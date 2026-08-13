"""The placement API. A decision over HTTP, for callers that are not a repo.

pilot delivers through a repository, which reaches every team that deploys
from git and no team that does not. An orchestrator, a CI pipeline, a
scheduler or a router cannot read a pull request. This is the same decision,
returned as JSON.

**The boundary this API does not cross.** It answers "where should this class
run" and returns a placement. It does not accept a prompt, does not return a
completion, and does not forward anything. A caller asking us to serve a
request gets a 400 with the reason, and that refusal is a route rather than an
omission so it cannot be mistaken for a missing feature.

That distinction is the whole architecture. A party that carries traffic
cannot credibly rank the placements it carries traffic for, so the API returns
decisions and the caller acts on them.

**No framework.** The standard library only, because a placement decision
should not oblige anyone to adopt a web stack, and because this has to run in
a customer's environment as easily as ours. Roughly two hundred lines against
a dependency that would be forty thousand.

    python -m berth.api --port 8080
    curl -s localhost:8080/v1/place -d '{"model":"llama3-8b","slo_ms":800,
      "concurrency":8,"prompt_tokens":512,"output_tokens":128}'
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from berth.place import decide
from berth.versus import ApiOffer, compare
from berth.workload import MODELS

# Deliberately small. Every field a caller must supply is one they already
# know, and nothing here requires understanding the estimator.
REQUIRED = ("model", "slo_ms", "concurrency", "prompt_tokens", "output_tokens")

# Requests per minute per client. Not security: a placement decision costs
# real compute to evaluate across a fleet, and an unbounded loop by a caller
# is indistinguishable from an attack until it is too late.
RATE_PER_MIN = 120


class Refused(ValueError):
    """The request is outside what this API does. Distinct from malformed."""


def _decide(body: dict) -> dict:
    """Resolve one placement decision.

    Returns the record the CLI prints, plus the ranking. Every number carries
    whether it rests on a measurement or a specification sheet, because a
    caller acting on this automatically has no chance to ask.
    """
    missing = [k for k in REQUIRED if k not in body]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if body["model"] not in MODELS:
        raise ValueError(f"unknown model {body['model']!r}. "
                         f"Known: {sorted(MODELS)}")

    rec = decide(
        workload_class=body.get("workload_class", "api-request"),
        model_key=body["model"],
        # Without an incumbent there is nothing to beat, so the first
        # accelerator in the fleet stands in and the caller sees the full
        # ranking rather than a comparison against nothing.
        incumbent=body.get("running_on") or next(iter(_fleet())),
        slo_metric=body.get("slo_metric", "p99_ttft_ms"),
        slo_bound_ms=float(body["slo_ms"]),
        batch=int(body["concurrency"]),
        prompt_tokens=int(body["prompt_tokens"]),
        output_tokens=int(body["output_tokens"]),
        bases=tuple(body.get("price_bases") or ("on-demand",)),
        interruption_tolerant=bool(body.get("interruption_tolerant", False)),
        prices=body.get("prices"))

    out = asdict(rec)
    # The caller needs to know whether to act, not just what ranked first.
    out["act"] = rec.clears_band
    out["why"] = (
        f"{rec.margin:.1%} better than {rec.incumbent}, against a "
        f"±{rec.band:.1%} band"
        if rec.clears_band else
        f"{rec.margin:.1%} improvement does not clear the ±{rec.band:.1%} "
        f"band plus the cost of moving. Do nothing.")
    return out


def _fleet():
    from berth.silicon import FLEET
    return FLEET


def _versus(body: dict) -> dict:
    """Self-host or API, on one axis.

    Takes the offers from the caller rather than holding a rate card. A price
    list we maintain would be stale the week after it shipped, and a wrong
    price is worse than no price because it looks authoritative.
    """
    offers = [ApiOffer(**o) for o in (body.get("api_offers") or [])]
    c = compare(model_key=body["model"],
                prompt_tokens=int(body["prompt_tokens"]),
                output_tokens=int(body["output_tokens"]),
                slo_bound_ms=float(body["slo_ms"]),
                requests_per_hour=float(body["requests_per_hour"]),
                api_offers=offers,
                concurrency=int(body.get("concurrency", 8)),
                engineering_cost_per_hour=float(
                    body.get("engineering_cost_per_hour", 0.0)))
    return asdict(c)


ROUTES = {
    "/v1/place": _decide,
    "/v1/versus": _versus,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "berth"
    sys_version = ""
    _seen: dict[str, list[float]] = {}

    def log_message(self, fmt, *args):
        """Quiet by default. A placement API logging every request to stderr
        is noise in someone else's logs, and the caller has their own."""

    # -- helpers ----------------------------------------------------------

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _rate_limited(self) -> bool:
        now = time.time()
        key = self.client_address[0]
        hits = [t for t in self._seen.get(key, []) if now - t < 60]
        hits.append(now)
        self._seen[key] = hits
        return len(hits) > RATE_PER_MIN

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        if self.path in ("/health", "/"):
            return self._send(200, {"ok": True, "service": "berth placement"})
        if self.path == "/v1/silicon":
            from berth.cli import MEASURED
            return self._send(200, {
                "silicon": {k: {"measured": k in MEASURED,
                                "price_hr": v.base_price_hr,
                                "mem_gb": v.mem_gb}
                            for k, v in _fleet().items()},
                "models": sorted(MODELS)})
        return self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self):
        if self._rate_limited():
            return self._send(429, {
                "error": "rate limited",
                "detail": f"{RATE_PER_MIN} requests per minute. A placement "
                          f"decision is evaluated across the fleet, so this "
                          f"is a cost limit rather than a security one."})

        # The refusal is a route rather than an absence, so a caller expecting
        # a proxy gets a reason instead of a 404 they might read as a missing
        # endpoint.
        if self.path in ("/v1/completions", "/v1/chat/completions",
                         "/v1/generate", "/generate"):
            return self._send(400, {
                "error": "this is not an inference endpoint",
                "detail": "berth returns placement decisions, not "
                          "completions. It never sits in the request path: a "
                          "party that carries traffic cannot credibly rank "
                          "the placements it carries traffic for. Ask "
                          "/v1/place where to run this workload, then send "
                          "the request there yourself.",
                "see": "https://docs.reckonresearch.com/pilot/"})

        fn = ROUTES.get(self.path)
        if fn is None:
            return self._send(404, {"error": f"no route {self.path}",
                                    "routes": sorted(ROUTES) + ["/v1/silicon"]})

        length = int(self.headers.get("content-length") or 0)
        if length > 1_000_000:
            return self._send(413, {"error": "body too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            return self._send(400, {"error": f"invalid JSON: {e}"})

        try:
            return self._send(200, fn(body))
        except SystemExit as e:
            # decide() raises SystemExit where no placement meets the bound.
            # That is a result rather than a failure, and it is the answer:
            # cost per served token here is undefined, not high.
            return self._send(422, {"error": "no feasible placement",
                                    "detail": str(e)})
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": str(e)})


def serve(port: int = 8080, host: str = "127.0.0.1"):
    """Bind to loopback by default.

    A placement API is an internal service. Binding to all interfaces by
    default is how something ends up on the public internet because nobody
    passed a flag.
    """
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"berth placement API on http://{host}:{port}")
    print("  POST /v1/place    where should this workload class run")
    print("  POST /v1/versus   self-host or API, on one axis")
    print("  GET  /v1/silicon  the fleet, and which cells are measured")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m berth.api")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback by default. Pass 0.0.0.0 deliberately.")
    a = p.parse_args(argv)
    return serve(a.port, a.host)


if __name__ == "__main__":
    raise SystemExit(main())
