"""sounding: berth's measurement instrument (the meter).

Takes a sounding of real silicon the way a navigator measures depth: drives a
live inference endpoint and records what actually happens, so berth's estimate
can be checked against ground truth. Turns rented GPU-hours into TraceRecord
JSONL.

Drives an OpenAI-compatible completion server (vLLM/SGLang) with streaming
requests, measures TTFT (time to first token) and TPOT (median inter-token
gap) per concurrency/context cell, and writes traces that `berth.calibrate`
consumes directly. Stdlib-only (http.client + threads) so the box you rent
needs nothing beyond Python.

Mock mode (--mock) generates measurements from the analytical model + noise:
the full pipeline (sweep -> JSONL -> calibrate -> MAPE) is testable with zero
GPUs, and doubles as the CI smoke test. On real hardware the ONLY thing that
changes is where the numbers come from.

Usage (real):
    python -m bench.sounding --base-url http://HOST:8000 --silicon h100-sxm \
        --model llama3-8b --model-id meta-llama/Meta-Llama-3-8B --out traces.jsonl
Usage (mock):
    python -m bench.sounding --mock --silicon h100-sxm --model llama3-8b --out traces.jsonl
"""

import argparse
import http.client
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from berth.silicon import FLEET
from berth.traces import TraceRecord
from berth.workload import MODELS, WorkloadSpec, profile

DEFAULT_GRID = {
    "batch": [1, 4, 8, 16, 32],
    "prompt": [512, 2048, 7680],
    "output": [128, 256],
    "reps": 3,
}


# ---------------------------------------------------------------- real path

def _auth_headers(api_key: str | None) -> dict:
    """Headers for a request. Bearer token when the endpoint requires one.

    vLLM and SGLang both take --api-key, and any endpoint reachable from
    outside a host almost certainly uses it. Without this the harness gets a
    401 and the only way forward is to redeploy the server without auth,
    which nobody running production traffic is going to do.
    """
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _stream_one(base_url: str, model_id: str, prompt_tokens: int, max_tokens: int,
                api_key: str | None = None):
    """One streaming completion; returns (ttft_s, tpot_s median). Fails loud."""
    u = urlparse(base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
    body = json.dumps({
        "model": model_id,
        # 'x ' pairs tokenize ~1 token each on Llama tokenizers; close enough
        # for load shaping — real prompt length is whatever the server reports.
        "prompt": "x " * prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},   # server-reported token counts
        "ignore_eos": True,          # vLLM: force exactly max_tokens of decode
    })
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body, _auth_headers(api_key))
    resp = conn.getresponse()
    if resp.status in (401, 403):
        raise SystemExit(
            f"server returned {resp.status}. The endpoint requires authentication "
            f"and no key was given. Pass --api-key, or set BERTH_API_KEY in the "
            f"environment. Do not redeploy the server without auth to work around "
            f"this: measuring a differently-configured server measures a different "
            f"thing.")
    if resp.status != 200:
        raise RuntimeError(f"server returned {resp.status}: {resp.read()[:200]!r}")
    # SSE framing note: http.client yields arbitrary byte *blocks*, not lines,
    # and events are \n\n-delimited. We buffer bytes and emit only complete
    # events. (Prior versions iterated `for raw in resp` and split on single
    # \n, which fragmented events and silently dropped every token.)
    ttft = None
    first_token_t = None
    last_token_t = None
    n_tokens = 0
    usage = None
    buf = b""
    done = False
    while not done:
        block = resp.read(1024)
        if not block:
            break
        buf += block
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                payload = line[len(b"data: "):]
                if payload == b"[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # `include_usage` attaches usage to EVERY chunk here, including
                # text-bearing ones. Capture it, but decide token-vs-skip on
                # whether a text delta is present (not on usage's presence).
                if chunk.get("usage"):
                    usage = chunk["usage"]
                ch = chunk.get("choices") or []
                if not ch or not ch[0].get("text"):
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                    first_token_t = now
                last_token_t = now
                n_tokens += 1
            if done:
                break
    conn.close()
    if ttft is None or n_tokens < 2:
        raise RuntimeError("stream produced <2 tokens; check model_id / server logs")
    # TPOT = decode wall-clock span / inter-token intervals. Timing the DECODE
    # PHASE and dividing by real token count is insensitive to how the server
    # batches SSE flushes. (Prior versions took median of inter-CHUNK gaps,
    # which measured buffer-flush latency ~20us, not the ~15ms/token decode
    # cadence — yielding TPOT ~1000x too fast and corrupting every downstream
    # bandwidth/KV/MFU fit.) Prefer the server's completion_tokens when present.
    served_tokens = usage.get("completion_tokens") if usage else None
    n_for_tpot = served_tokens if served_tokens and served_tokens >= 2 else n_tokens
    tpot = (last_token_t - first_token_t) / max(1, n_for_tpot - 1)
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    return ttft, tpot, prompt_tokens


def measure_cell(base_url, model_id, batch, prompt, output, api_key=None):
    """Fire `batch` concurrent streams; return (ttft_ms, tpot_ms, prompt_toks)."""
    with ThreadPoolExecutor(max_workers=batch) as ex:
        results = list(ex.map(
            lambda _: _stream_one(base_url, model_id, prompt, output, api_key),
            range(batch),
        ))
    ttfts = [r[0] for r in results]
    tpots = [r[1] for r in results]
    reported = [r[2] for r in results if r[2] is not None]
    prompt_toks = round(statistics.median(reported)) if reported else prompt
    return statistics.median(ttfts) * 1e3, statistics.median(tpots) * 1e3, prompt_toks


# ---------------------------------------------------------------- mock path

def measure_cell_mock(silicon, model_name, batch, prompt, output, rng):
    """Analytical truth + 5% lognormal noise: validates the pipeline, not the model."""
    from berth.estimate import estimate
    hw = FLEET[silicon]
    sig = profile(WorkloadSpec(model=MODELS[model_name], target_batch=batch,
                               avg_prompt_tokens=prompt, avg_output_tokens=output))
    e = estimate(sig, hw, hw.base_price_hr)
    if not e.feasible:
        return None
    n = lambda: math.exp(rng.gauss(0.0, 0.05))
    return e.ttft_ms * n(), e.tpot_ms * n()


# ------------------------------------------------------------------- sweep

def run_sweep(args) -> list[TraceRecord]:
    rng = random.Random(args.seed)
    traces: list[TraceRecord] = []
    total = 0
    # Randomize cell order: fixed grid order confounds thermal/clock drift
    # with batch/context position. Shuffle is seeded -> reproducible.
    cells = [(b, p, o, r)
             for b in args.grid["batch"]
             for p in args.grid["prompt"]
             for o in args.grid["output"]
             for r in range(args.grid["reps"])]
    rng.shuffle(cells)
    warmed: set[tuple] = set()   # cell keys whose cold-cache run was discarded
    for batch, prompt, output, _rep in cells:
                    if args.mock:
                        m = measure_cell_mock(args.silicon, args.model,
                                              batch, prompt, output, rng)
                        if m is None:
                            continue
                        ttft_ms, tpot_ms = m
                    else:
                        # Warm-up rep is measured but discarded (cold caches).
                        ttft_ms, tpot_ms, prompt_actual = measure_cell(
                            args.base_url, args.model_id, batch, prompt, output,
                            getattr(args, "api_key", None))
                        prompt = prompt_actual   # record truth, not the request
                        # Discard the first chronological execution of each
                        # cell (cold caches, CUDA-graph capture). The shuffled
                        # rep label is NOT chronology — keying on rep==0 after
                        # randomization discards a random rep instead.
                        key = (batch, prompt, output)
                        if key not in warmed and args.grid["reps"] > 1:
                            warmed.add(key)
                            continue
                    traces.append(TraceRecord(
                        silicon=args.silicon, model_name=args.model,
                        batch=batch, avg_prompt_tokens=prompt,
                        avg_output_tokens=output,
                        measured_ttft_ms=ttft_ms, measured_tpot_ms=tpot_ms,
                        t=total / 1000.0,
                        w_bytes=getattr(args, "weight_bytes", 2.0),
                        kv_bytes=getattr(args, "kv_bytes", 2.0),
                        source="mock" if args.mock else "measured",
                    ))
                    total += 1
                    print(f"[{total}] b={batch} p={prompt} o={output} "
                          f"TTFT={ttft_ms:.0f}ms TPOT={tpot_ms:.1f}ms")
    return traces


# v1: original. v2: TWO different v2s were minted in parallel branches, one
# adding w_bytes/kv_bytes and one adding source. Neither is a superset of the
# other, so v2 on disk is ambiguous and the loader below back-fills whichever
# half is absent. v3 is the reconciled schema carrying both.
SCHEMA_VERSION = 3
SUPPORTED_SCHEMAS = (1, 2, 3)


def save_jsonl(traces: list[TraceRecord], path: str) -> None:
    with open(path, "w") as f:
        for tr in traces:
            f.write(json.dumps({"schema": SCHEMA_VERSION, **tr.__dict__}) + "\n")


def load_jsonl(path: str) -> list[TraceRecord]:
    """Versioned loader. Unversioned lines are treated as v1 (pre-versioning
    files from this repo only); unknown versions fail loud — silently
    misparsing external contributions is how an index corrupts."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            v = d.pop("schema", 1)
            if v not in SUPPORTED_SCHEMAS:
                raise ValueError(f"trace schema v{v} unsupported (loader is v{SCHEMA_VERSION})")
            if v < 3:
                # v1 predates both additions. v2 carries exactly one of them,
                # and which one depends on the branch that wrote it, so default
                # whatever is missing rather than guessing at the writer.
                # bf16 and measured are the right defaults: every pre-v3 file in
                # this repo is hardware at bf16. Contributions are held to a
                # stricter rule (bench.check_contributed requires the field
                # explicitly), because an outside file has unknown origin.
                d.setdefault("w_bytes", 2.0)
                d.setdefault("kv_bytes", 2.0)
                d.setdefault("source", "measured")
            out.append(TraceRecord(**d))
    return out


def provenance_of(traces: list[TraceRecord]) -> str:
    """Sole provenance of a trace set. A file is measurement or it is
    rehearsal, never both: a mixed set has no defensible label, and the
    inheritance rule would force the whole thing down to mock anyway."""
    kinds = {t.source for t in traces}
    if len(kinds) > 1:
        raise SystemExit(
            f"refusing a trace set mixing {sorted(kinds)}. Split them; a mixed "
            "set has no provenance, so no number derived from it has one either.")
    return kinds.pop() if kinds else "measured"


def verify_served_model(base_url: str, model_id: str,
                        api_key: str | None = None) -> str:
    """Confirm the endpoint is serving the model we are about to attribute to.

    WHY this is not optional. `--model` is a berth registry key (parameter
    count, layers, attention family) and `--model-id` is whatever the server
    loaded. Nothing else connects them. Point the harness at a server running
    Qwen while passing `--model llama3-8b` and every request succeeds, every
    timing is real, and all 90 cells are attributed to the wrong parameter
    count. The roofline then inverts against the wrong weights and the fitted
    mfu/bw_eff silently absorb the discrepancy.

    That failure is invisible: no exception, no warning, plausible numbers.
    It is the last path in this harness by which a measurement can be wrong
    without anyone noticing, which is exactly the class of error the corpus
    cannot survive.

    Returns the served model id on success. Raises SystemExit on mismatch.
    """
    u = urlparse(base_url)
    conn = None
    try:
        # Construction inside the try: a refused connection raises here, not at
        # request time, and a raw traceback is not a useful thing to hand
        # someone who is paying for a GPU by the minute.
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
        conn.request("GET", "/v1/models", headers=_auth_headers(api_key))
        resp = conn.getresponse()
        if resp.status in (401, 403):
            raise SystemExit(
                f"GET /v1/models returned {resp.status}. The endpoint requires "
                f"authentication; pass --api-key or set BERTH_API_KEY.")
        if resp.status != 200:
            raise SystemExit(
                f"GET /v1/models returned {resp.status}. Cannot confirm what the "
                f"endpoint is serving, so refusing to attribute measurements to "
                f"--model-id {model_id!r}. Check --base-url points at a running "
                f"OpenAI-compatible server.")
        served = [m.get("id") for m in json.loads(resp.read()).get("data", [])]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"could not reach {base_url}/v1/models: {exc}. The server must be up "
            f"before the sweep starts; a sweep against a dead endpoint wastes the "
            f"rental and produces nothing.") from exc
    finally:
        if conn is not None:
            conn.close()

    if model_id not in served:
        raise SystemExit(
            f"--model-id {model_id!r} is not served by {base_url}.\n"
            f"  serving: {served or '(nothing)'}\n"
            f"Every request would still succeed and every timing would be real, "
            f"but the cells would be attributed to weights the server never ran. "
            f"Pass one of the ids above, or start the server on {model_id!r}.")
    return model_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silicon", required=True, choices=sorted(FLEET))
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--base-url", help="OpenAI-compatible server (real mode)")
    p.add_argument("--model-id", help="served model id, e.g. meta-llama/... (real mode)")
    p.add_argument("--mock", action="store_true", help="analytical truth + noise, no GPU")
    p.add_argument("--out", default="traces.jsonl")
    p.add_argument("--seed", type=int, default=0)
    # Quant of the served model, bytes per weight/KV element (bf16=2, fp8/int8=1,
    # fp4/int4=0.5). Recorded per trace so a fp8 cell is inverted as fp8, never
    # silently compared to bf16 in the premium table.
    p.add_argument("--weight-bytes", type=float, default=2.0)
    p.add_argument("--kv-bytes", type=float, default=2.0)
    p.add_argument("--api-key", default=os.environ.get("BERTH_API_KEY"),
                   help="bearer token if the endpoint requires one "
                        "(default: $BERTH_API_KEY)")
    args = p.parse_args()
    if not args.mock and not (args.base_url and args.model_id):
        p.error("real mode requires --base-url and --model-id (or pass --mock)")
    if not args.mock:
        # Fail here, before renting time is spent, rather than after 90 cells.
        served = verify_served_model(args.base_url, args.model_id, args.api_key)
        print(f"endpoint serving {served}, attributing to --model {args.model}")
        # --silicon cannot be verified over an OpenAI-compatible API: the
        # endpoint does not report what it runs on. It is an assertion by the
        # operator, and a wrong one produces ninety cells of real timings
        # attributed to hardware that never ran them. Nothing downstream can
        # detect that, so the only defence is saying so out loud.
        print(f"WARNING: --silicon {args.silicon} is declared, not detected. "
              f"Nothing here verifies this box is an {args.silicon}. If it is "
              f"not, every cell will be attributed to the wrong hardware and "
              f"no later check will catch it. Confirm with nvidia-smi before "
              f"the sweep starts.")
    args.grid = DEFAULT_GRID
    traces = run_sweep(args)
    save_jsonl(traces, args.out)
    print(f"wrote {len(traces)} traces -> {args.out}")


if __name__ == "__main__":
    main()
