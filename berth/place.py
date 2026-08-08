"""The placement decision, as a record a contract can reference.

`berth estimate` answers what a workload costs on a given accelerator. That is
an estimate, not a decision. A decision names the placement recommended, the
alternative it beat, the margin between them, whether that margin clears the
uncertainty on it, and how much of the answer rests on measurement rather than
on a datasheet.

The distinction matters commercially. A savings-share contract is written
against a decision, not an estimate: both parties need to point at the same
document and agree what was recommended, when, and on what basis. An estimate
that changes when the corpus moves is correct behaviour; a decision that
changes silently is a dispute.

This module is the control plane's first component. It never touches a
request and never moves traffic. It produces a document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from berth.agent import Decision
from berth.estimate import KV_PRESSURE_WARN, estimate
from berth.silicon import FLEET
from berth.workload import MODELS, WorkloadSpec, profile

# Cells measured against hardware, with the published error on each. Anything
# absent is spec-sheet arithmetic and is labelled as such on every decision
# that rests on it, because a prior has been observed above 40 percent error
# in this corpus and a buyer moving production traffic deserves to know which
# kind of number they are looking at.
def _load_bands():
    """Published error per measured cell, from data rather than from code.

    The model is open source and the measurements are the product. Keeping
    the bands in a data file rather than a literal is what lets the published
    corpus and a licensed one be different files against the same estimator,
    and it means adding a cell does not require a release.

    Falls back to the published set, so the package works standalone.
    """
    import json
    import os
    path = os.environ.get("BERTH_CORPUS_BANDS")
    if path:
        with open(path) as f:
            return {(c["silicon"], c["model"]): c["band"] for c in json.load(f)}
    return {
        ("l40s", "llama3-8b"): 0.106,
        ("h100-pcie", "llama3-8b"): 0.042,
        ("h100-sxm", "llama3-8b"): 0.042,
        ("mi300x", "llama3-8b"): 0.106,
        ("a100-80g", "qwen3-30b-a3b"): 0.376,
    }


MEASURED_CELLS = _load_bands()

# Applied where no cell covers the placement. Not a confidence interval: an
# admission that the error is unbounded, set at a level that suppresses
# proposals resting on nothing.
PRIOR_BAND = 0.40


@dataclass
class Candidate:
    """One placement, evaluated."""

    silicon: str
    cost_per_mtok: float
    ttft_ms: float
    tpot_ms: float
    devices: int
    feasible: bool
    kv_pressure: float
    measured: bool
    band: float
    excluded_reason: str | None = None


@dataclass
class PlacementRecord:
    """The document. Referenced by a contract, carried on a receipt."""

    workload_class: str
    model_key: str
    slo_metric: str
    slo_bound_ms: float
    batch: int
    prompt_tokens: int
    output_tokens: int
    incumbent: str
    recommended: str
    candidates: list[dict]
    margin: float
    band: float
    clears_band: bool
    measured_cells: int
    prior_cells: int
    excluded: list[dict] = field(default_factory=list)
    created_utc: str = ""
    provenance: dict = field(default_factory=dict)

    def to_decision(self) -> Decision:
        """The form the agent consumes."""
        by = {c["silicon"]: c for c in self.candidates}
        rec, inc = by.get(self.recommended), by.get(self.incumbent)
        return Decision(
            recommended=self.recommended,
            incumbent=self.incumbent,
            recommended_cost=rec["cost_per_mtok"] if rec else 0.0,
            incumbent_cost=inc["cost_per_mtok"] if inc else 0.0,
            confidence_band=self.band,
            measured_cells=self.measured_cells,
            prior_cells=self.prior_cells,
            feasible=bool(inc and inc["feasible"]),
            reason_infeasible=(inc.get("excluded_reason") if inc else None),
        )


def decide(*, workload_class: str, model_key: str, incumbent: str,
           slo_metric: str = "p99_ttft_ms", slo_bound_ms: float = 1000.0,
           batch: int = 8, prompt_tokens: int = 512, output_tokens: int = 128,
           fleet=None, prices=None) -> PlacementRecord:
    """Evaluate every placement in the fleet and record the decision.

    A placement missing the latency bound is excluded rather than ranked
    cheaply. Its cost per compliant token is not high, it is undefined,
    because a first token arriving after the deadline has not participated in
    the interaction it was generated for.

    A placement whose key-value cache will not fit one card is excluded too,
    with the reason stated. The server would preempt and recompute, and
    recompute time appears in no term of the model, so estimating through it
    would be predicting a number for a regime the model does not describe.
    """
    if model_key not in MODELS:
        raise SystemExit(f"unknown model {model_key!r}")
    fleet = fleet or FLEET
    prices = prices or {}

    sig = profile(WorkloadSpec(model=MODELS[model_key],
                               avg_prompt_tokens=prompt_tokens,
                               avg_output_tokens=output_tokens,
                               target_batch=batch))

    cands, excluded = [], []
    for key, hw in fleet.items():
        price = prices.get(key, hw.base_price_hr)
        e = estimate(sig, hw, price)
        band = MEASURED_CELLS.get((key, model_key), PRIOR_BAND)
        c = Candidate(
            silicon=key,
            cost_per_mtok=e.cost_per_mtok if e.feasible else float("inf"),
            ttft_ms=e.ttft_ms, tpot_ms=e.tpot_ms, devices=e.n_devices,
            feasible=bool(e.feasible), kv_pressure=e.kv_pressure,
            measured=(key, model_key) in MEASURED_CELLS, band=band)

        if not c.feasible:
            c.excluded_reason = "the estimator declares this placement infeasible"
        elif e.ttft_ms > slo_bound_ms:
            c.excluded_reason = (f"first-token latency {e.ttft_ms:.0f} ms "
                                 f"exceeds the {slo_bound_ms:.0f} ms bound, so "
                                 f"cost per compliant token is undefined")
        elif c.kv_pressure == float("inf"):
            c.excluded_reason = ("the weights alone exceed one card's memory, "
                                 "so there is no room for a cache at all")
        elif c.kv_pressure > KV_PRESSURE_WARN:
            c.excluded_reason = (f"key-value cache needs {c.kv_pressure:.2f} of "
                                 f"one card's free memory; the server will "
                                 f"preempt and recompute, and recompute time "
                                 f"appears in no term of the model")

        (excluded if c.excluded_reason else cands).append(c)

    cands.sort(key=lambda c: c.cost_per_mtok)
    if not cands:
        raise SystemExit(
            f"no placement in the fleet meets a {slo_bound_ms:.0f} ms bound "
            f"for this workload. Cost per compliant token here is not high, "
            f"it is undefined, and that is the finding.")

    best = cands[0]
    inc = next((c for c in cands + excluded if c.silicon == incumbent), None)
    if inc is None:
        raise SystemExit(f"incumbent {incumbent!r} is not in the fleet")

    margin = ((inc.cost_per_mtok - best.cost_per_mtok) / inc.cost_per_mtok
              if inc.cost_per_mtok not in (0, float("inf")) else 0.0)
    # The band on a comparison is the wider of the two, because a difference
    # is only as certain as its least certain side.
    band = max(best.band, inc.band)

    return PlacementRecord(
        workload_class=workload_class, model_key=model_key,
        slo_metric=slo_metric, slo_bound_ms=slo_bound_ms, batch=batch,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        incumbent=incumbent, recommended=best.silicon,
        candidates=[asdict(c) for c in cands],
        excluded=[asdict(c) for c in excluded],
        margin=margin, band=band, clears_band=margin > band,
        measured_cells=sum(1 for c in cands if c.measured),
        prior_cells=sum(1 for c in cands if not c.measured),
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        provenance={
            "cost": "DERIVED from a closed-form roofline plus a queueing term",
            "band": "MEASURED where a cell exists, otherwise an unbounded prior",
            "prices": "CONFIG, list rate, on-demand, single tenant",
            "feasibility": "DERIVED from single-device memory, not the chosen layout",
        })


def render(rec: PlacementRecord) -> str:
    """Human-readable. The record is authoritative; this is for reading."""
    out = [
        f"PLACEMENT DECISION   {rec.workload_class}",
        "=" * 66,
        f"  model      {rec.model_key}",
        f"  workload   concurrency {rec.batch}, {rec.prompt_tokens} prompt, "
        f"{rec.output_tokens} output",
        f"  bound      {rec.slo_metric} under {rec.slo_bound_ms:.0f} ms",
        "",
        f"  {'':<14}{'$/Mtok':>10}{'ttft':>9}{'tpot':>8}{'dev':>5}  basis",
    ]
    for c in rec.candidates:
        mark = "->" if c["silicon"] == rec.recommended else \
               "  " if c["silicon"] != rec.incumbent else "= "
        basis = "MEASURED" if c["measured"] else f"prior, ±{c['band']:.0%}"
        out.append(f"{mark}{c['silicon']:<14}{c['cost_per_mtok']:>10.3f}"
                   f"{c['ttft_ms']:>8.0f}m{c['tpot_ms']:>7.1f}m{c['devices']:>5}"
                   f"  {basis}")
    if rec.excluded:
        out += ["", "  excluded"]
        for c in rec.excluded:
            out.append(f"    {c['silicon']:<14}{c['excluded_reason']}")
    out += [
        "",
        f"  incumbent    {rec.incumbent}",
        f"  recommended  {rec.recommended}",
        f"  margin       {rec.margin:.1%} against a band of ±{rec.band:.1%}",
        f"  verdict      {'act' if rec.clears_band else 'inside the noise, do nothing'}",
        f"  basis        {rec.measured_cells} measured, {rec.prior_cells} prior",
    ]
    return "\n".join(out)
