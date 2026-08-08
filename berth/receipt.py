"""Settlement: compare two legs, compute the delta, emit a conforming receipt.

The billing event is the measurement. This module turns a closed period into
the artifact that justifies an invoice, or into the artifact that explains why
there is not one.

Every figure carries where it came from. A receipt whose numbers cannot be
recomputed from published traces is a claim, and the whole argument of this
company is that a claim is not a measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from berth.holdout import Declaration, LegObservations, ProtocolError, stationarity, wilson_interval


@dataclass
class Settlement:
    """The arithmetic of one period, before it becomes a document."""

    compliant_work_mtok: float
    baseline_cost_per_mtok: float
    treatment_cost_per_mtok: float
    difference_per_mtok: float
    gross_saving: float
    holdout_cost: float
    net_saving: float
    carry_forward_applied: float
    billable_saving: float
    share: float
    invoice: float
    invoice_conservative: float
    customer_retains: float


def settle(declaration: Declaration, baseline: LegObservations,
           treatment: LegObservations, carry_forward: float = 0.0) -> Settlement:
    """Compute what is owed, if anything.

    The holdout runs on the worse placement by design. It costs the customer
    real money and that cost is deducted from the saving before the share is
    applied, because netting it silently would make every receipt an
    overstatement.

    A negative period carries forward. Without that the arrangement captures a
    share of every gain and none of any loss, which is an asymmetry a
    counterparty is right to refuse, and it costs nothing in the case where
    the recommendation was correct.
    """
    if carry_forward < 0:
        raise ProtocolError("carry_forward is a loss to offset and is positive")

    bc = baseline.cost_per_mtok
    tc = treatment.cost_per_mtok
    diff = bc - tc

    work_mtok = (baseline.served_tokens + treatment.served_tokens) / 1e6
    holdout_mtok = baseline.served_tokens / 1e6

    gross = work_mtok * diff
    holdout_cost = holdout_mtok * diff
    net = gross - holdout_cost

    billable = max(0.0, net - carry_forward)
    invoice = billable * declaration.share

    # Bill at the conservative end of the compliance interval. The upper bound
    # on compliance produces the lower bound on cost per served token, so it
    # produces the smaller saving.
    _, b_hi = wilson_interval(baseline.compliant_requests, baseline.requests)
    _, t_hi = wilson_interval(treatment.compliant_requests, treatment.requests)
    b_cons = (baseline.spend / (baseline.served_tokens / 1e6
                                * (b_hi / baseline.compliance_rate))
              if baseline.served_tokens and baseline.compliance_rate else bc)
    t_cons = (treatment.spend / (treatment.served_tokens / 1e6
                                 * (t_hi / treatment.compliance_rate))
              if treatment.served_tokens and treatment.compliance_rate else tc)
    cons_diff = b_cons - t_cons
    cons_net = work_mtok * cons_diff - holdout_mtok * cons_diff
    cons_invoice = max(0.0, cons_net - carry_forward) * declaration.share

    return Settlement(
        compliant_work_mtok=work_mtok,
        baseline_cost_per_mtok=bc,
        treatment_cost_per_mtok=tc,
        difference_per_mtok=diff,
        gross_saving=gross,
        holdout_cost=holdout_cost,
        net_saving=net,
        carry_forward_applied=min(carry_forward, max(0.0, net)),
        billable_saving=billable,
        share=declaration.share,
        invoice=invoice,
        invoice_conservative=min(invoice, cons_invoice),
        customer_retains=billable - invoice,
    )


@dataclass
class Receipt:
    """A conforming record for one measurement period."""

    declaration: dict
    baseline: dict
    treatment: dict
    settlement: dict
    checks: dict
    warmup_excluded_requests: int
    evaluated_quality_fraction: float
    serving_stack: dict
    trace_pointer: str
    billable: bool
    reason_not_billable: str | None
    provenance: dict = field(default_factory=dict)
    created_utc: str = ""

    def to_json(self, indent=2) -> str:
        d = asdict(self)
        d["created_utc"] = self.created_utc or datetime.now(
            UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return json.dumps(d, indent=indent, sort_keys=False)


def build_receipt(declaration: Declaration, baseline: LegObservations,
                  treatment: LegObservations, *, trace_pointer: str,
                  serving_stack: dict, warmup_excluded: int = 0,
                  evaluated_quality_fraction: float = 0.02,
                  carry_forward: float = 0.0,
                  breaker_tripped: bool = False) -> Receipt:
    """Turn a closed period into the document that settles it.

    A period is billable only when it passes the blocking stationarity checks
    and the circuit breaker did not trip. Every other outcome still produces a
    receipt, because a period that cannot be billed is a result and publishing
    it is what makes the ones that can be billed credible.
    """
    checks, blocking = stationarity(baseline, treatment)
    s = settle(declaration, baseline, treatment, carry_forward)

    reason = None
    if breaker_tripped:
        # Voided rather than counted as a loss: a placement that never held
        # the bound was never a measurement, so it does not carry forward.
        reason = ("circuit breaker tripped: the treatment leg breached the "
                  "declared bound, the placement reverted immediately, and "
                  "the period is void rather than a loss")
    elif blocking:
        failed = [k for k, (_v, ok) in checks.items() if not ok]
        reason = (f"stationarity failed on {failed}. The traffic reaching the "
                  f"two legs was not the same traffic, so the difference "
                  f"between them is not attributable to the placement")
    elif s.net_saving <= 0:
        reason = (f"measured saving is {s.net_saving:,.0f}, so no invoice. The "
                  f"loss carries forward against the next period and the "
                  f"placement reverts at the boundary")

    b_lo, b_hi = wilson_interval(baseline.compliant_requests, baseline.requests)
    t_lo, t_hi = wilson_interval(treatment.compliant_requests, treatment.requests)

    return Receipt(
        declaration=asdict(declaration),
        baseline={**asdict(baseline), "cost_per_mtok": baseline.cost_per_mtok,
                  "compliance_rate": baseline.compliance_rate,
                  "compliance_interval": [b_lo, b_hi]},
        treatment={**asdict(treatment), "cost_per_mtok": treatment.cost_per_mtok,
                   "compliance_rate": treatment.compliance_rate,
                   "compliance_interval": [t_lo, t_hi]},
        settlement=asdict(s),
        checks={k: {"value": v, "pass": ok} for k, (v, ok) in checks.items()},
        warmup_excluded_requests=warmup_excluded,
        evaluated_quality_fraction=evaluated_quality_fraction,
        serving_stack=serving_stack,
        trace_pointer=trace_pointer,
        billable=reason is None,
        reason_not_billable=reason,
        provenance={
            "spend": "MEASURED, customer's metered cost at contracted rate",
            "served_tokens": "MEASURED, compliant output only",
            "compliance": "MEASURED, joint sample over both legs",
            "interval": "DERIVED, Wilson score at 95 percent",
            "saving": "DERIVED from the two legs, holdout cost deducted",
            "assignment": "DETERMINISTIC from the published seed",
        },
    )


def render(receipt: Receipt) -> str:
    """Human-readable form. The JSON is authoritative; this is for reading."""
    s = receipt.settlement
    b, t = receipt.baseline, receipt.treatment
    d = receipt.declaration
    out = [
        f"PLACEMENT RECEIPT   {d['workload_class']}",
        "=" * 62,
        f"  baseline    {d['baseline_placement']}",
        f"  treatment   {d['treatment_placement']}",
        f"  bound       {d['slo_metric']} under {d['slo_bound_ms']:.0f} ms",
        f"  quality     {d['quality_floor']}",
        f"  holdout     {d['holdout_fraction']:.1%}, seed published in advance",
        "",
        f"  {'':<22}{'baseline':>18}{'treatment':>18}",
        f"  {'spend':<22}{b['spend']:>18,.0f}{t['spend']:>18,.0f}",
        f"  {'compliant MSVT':<22}{b['served_tokens']/1e6:>18,.0f}"
        f"{t['served_tokens']/1e6:>18,.0f}",
        f"  {'compliance':<22}{b['compliance_rate']:>17.1%}{t['compliance_rate']:>18.1%}",
        f"  {'cost per Mtok':<22}{b['cost_per_mtok']:>18,.2f}{t['cost_per_mtok']:>18,.2f}",
        "",
        f"  compliant work            {s['compliant_work_mtok']:>12,.0f} MSVT",
        f"  difference per Mtok       {s['difference_per_mtok']:>12,.2f}",
        f"  gross saving              {s['gross_saving']:>12,.0f}",
        f"  holdout cost              {s['holdout_cost']:>12,.0f}",
        f"  net saving                {s['net_saving']:>12,.0f}",
    ]
    if s["carry_forward_applied"]:
        out.append(f"  carry-forward applied     {s['carry_forward_applied']:>12,.0f}")
    out += [
        f"  share at {s['share']:.0%}               {s['invoice']:>12,.0f}",
        f"  invoiced, conservative    {s['invoice_conservative']:>12,.0f}",
        f"  customer retains          {s['customer_retains']:>12,.0f}",
        "",
        "  checks",
    ]
    for k, v in receipt.checks.items():
        out.append(f"    {k:<22}{v['value']:>10.4f}   {'pass' if v['pass'] else 'FAIL'}")
    out += ["", f"  warm-up excluded  {receipt.warmup_excluded_requests:,} requests",
            f"  quality evaluated on {receipt.evaluated_quality_fraction:.1%} of both legs",
            f"  traces            {receipt.trace_pointer}"]
    if not receipt.billable:
        out += ["", f"  NOT BILLABLE: {receipt.reason_not_billable}"]
    return "\n".join(out)
