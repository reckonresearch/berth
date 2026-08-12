"""Precompute the placement grid that /index/ renders.

WHY precompute rather than reimplement the physics in JavaScript. The page and
the CLI must never disagree. Reimplementing the roofline in JS creates a second
copy of the model that drifts silently, and a leaderboard whose numbers differ
from the tool it advertises is worse than no leaderboard. This calls the same
estimator the CLI calls and freezes the result.

Run from the berth repo root:
    python3 gen_index_data.py > index_data.json

Then copy index_data.json into the site repo at assets/index_data.json.
"""

import json
import math
import sys

from berth.cli import MEASURED
from berth.estimate import estimate
from berth.silicon import FLEET
from berth.workload import MODELS, WorkloadSpec, profile

# Axes the page exposes. Kept coarse on purpose: every extra point multiplies
# the payload, and the argument the page makes is visible at this resolution.
BATCH = [1, 4, 8, 16, 32, 64]
PROMPT = [512, 2048, 8192, 32768]
OUTPUT = [16, 128, 512]

def fin(x, nd):
    """Round, or None if the value is not finite."""
    if x is None or not math.isfinite(x):
        return None
    return round(x, nd)


PREFERRED = ["llama3-8b", "llama3-70b", "qwen3-30b-a3b", "mixtral-8x7b",
             "qwen3-235b-a22b", "deepseek-v3", "deepseek-v2-lite"]
MODELKEYS = [k for k in PREFERRED if k in MODELS] or sorted(MODELS)

grid = {}
for mk in MODELKEYS:
    for b in BATCH:
        for p in PROMPT:
            for o in OUTPUT:
                sig = profile(WorkloadSpec(model=MODELS[mk], avg_prompt_tokens=p,
                                           avg_output_tokens=o, target_batch=b))
                rows = []
                for sk, sp in FLEET.items():
                    e = estimate(sig, sp, sp.base_price_hr)
                    # Infeasible placements come back with infinite cost.
                    # Python's json writes `Infinity`, which is not valid JSON
                    # and which JSON.parse rejects outright, taking the whole
                    # page down rather than the one row. Encode it as null and
                    # let the client treat it as infeasible.
                    rows.append({
                        "s": sk,
                        "c": fin(e.cost_per_mtok, 4),
                        "ttft": fin(e.ttft_ms, 1),
                        "tpot": fin(e.tpot_ms, 2),
                        "tps": fin(e.tokens_per_s, 1),
                        "n": e.n_devices,
                        "f": bool(e.feasible),
                        "m": sk in MEASURED,
                    })
                grid[f"{mk}|{b}|{p}|{o}"] = rows

json.dump({
    "note": "precomputed by gen_index_data.py from the same estimator berth's "
            "CLI uses. Regenerate whenever the fleet, the model registry or "
            "any physics term changes.",
    "axes": {"model": MODELKEYS, "batch": BATCH, "prompt": PROMPT,
             "output": OUTPUT},
    "silicon": {k: {"price_hr": v.base_price_hr, "measured": k in MEASURED}
                for k, v in FLEET.items()},
    "grid": grid,
}, sys.stdout, separators=(",", ":"))
