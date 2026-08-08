# The Placement Holdout Protocol

**Version 0.1, draft. Reckon Research.**

A method for establishing what a workload *would have* cost, so that a change
in placement can be billed on the difference.

---

## Why this document exists

A savings-share contract turns on one number that is unobservable: what the
customer would have spent had they not changed anything. Everything else in
the arrangement is bookkeeping.

Energy performance contracting had the same problem and solved it in 1996 with
a measurement and verification protocol. Before it, savings-share contracts
existed and were litigated. After it, performance contracting became an
industry. The protocol was worth more than any individual meter, because a
meter tells you what happened and a protocol tells you what would have.

This is that document for inference placement. It is published openly, under
the same terms as the unit specification, because a method only one party may
use is not a method, it is a claim.

**What it is not.** It is not an experiment design for research. The standard
here is lower and different: it must produce a number two commercial parties
will both sign, under conditions where one of them is paying and the other is
being paid. That constrains it toward simplicity, pre-commitment, and
falsifiability rather than toward statistical power.

---

## 1. Declaration, before anything moves

Nothing in this protocol is valid unless the following are recorded and
countersigned **before** the first request is routed differently.

**1.1 The baseline placement.** The silicon, count, serving stack and version,
quantization, and the launch configuration in force today. Stated as the
customer's current policy, not as a reconstruction.

**1.2 The service level.** The bound being held: a percentile, a metric, and a
value. For example p99 first-token latency under 800 ms. If more than one
bound applies, all of them, because a saving obtained by breaching an
unstated bound is not a saving.

**1.3 The quality floor.** The acceptance test a response must pass to count.
Declared by the customer, evaluated by the customer or an agreed third party.
Reckon does not define what quality means for a workload. Output failing the
floor is zero work that was paid for, not discounted work.

**1.4 The workload class.** The traffic this contract covers, identified by
route, model, or tenant, and separable in the customer's own logs. A contract
that cannot be scoped in the logs cannot be verified.

**1.5 The holdout fraction and duration.** See section 2.

**1.6 The assignment seed.** Pre-committed and published. This is the specific
defence against selective measurement and it is the rule both parties will be
tempted to break.

Any change to 1.1 through 1.4 during a measurement period **voids the period**.
It does not adjust it. A baseline that moves during measurement is not a
baseline.

---

## 2. The holdout

A declared fraction of production traffic continues to run on the baseline
placement for the duration of the contract.

**2.1 It runs on the worse placement by design.** That is the point, and it
costs the customer real money. The cost is calculated, disclosed in the
contract, and deducted from the measured saving before billing. Hiding it
would make every number in the receipt an overstatement.

**2.2 Fraction.** Default 5 percent. The floor is whatever produces at least
1,000 requests per measurement period on the holdout leg, and at least 30
observations above the declared percentile bound. Below that the percentile is
not estimable and the protocol requires the interval to be reported rather
than the point.

**2.3 Assignment happens per request, at admission, by hashing a request
identifier with the pre-committed seed**, using SHA-256 truncated to 64 bits.
The identifier must be a UUID or an equivalently unstructured value.
Sequential identifiers, tenant-prefixed identifiers and timestamps all carry
structure that survives a weak hash and produces structured assignment; where
the customer's identifiers carry structure, the reference implementation
prepends the seed before hashing, which removes it. Not per tenant, not per session, not
per hour. Any coarser assignment lets a diurnal or per-customer pattern land
disproportionately on one leg, and the resulting difference is a property of
the traffic rather than of the placement.

**2.4 Assignment must not be visible to the serving path before the response
is produced.** If a request can be identified as holdout on the way in, it can
be treated differently, and the comparison measures the treatment rather than
the placement.

**2.5 Duration.** A measurement period is 30 days or 100,000 requests on the
holdout leg, whichever comes later. Shorter periods are permitted for a pilot
and are labelled as such on the receipt.

**2.6 Warm-up, excluded from measurement.** A newly started placement has cold
caches, unloaded weights and uncaptured compiled graphs. Comparing a warm
baseline against a cold treatment measures the start, not the placement.

The warm-up ends when the treatment leg's cost per served token is stable
within 5 percent across two consecutive intervals of at least 500 requests
each. Requests before that point are excluded from both legs and the excluded
count appears on the receipt. If stability is not reached within 10,000
requests, the period does not start and the configuration is reviewed.

**2.7 Balance check.** Realised assignment must land within 1 percentage point
of the declared fraction over the period. Outside that, the period is flagged
and the assignment function is examined before anything is billed. A hash that
does not distribute evenly over the customer's identifier format is a defect
in the instrument, not a result.

---

## 3. What is compared

For each leg, over the same wall-clock window:

    C  =  total spend on the placement
          -----------------------------------------
          served tokens delivered inside the bound
                and above the quality floor

**3.1 Spend** is the metered cost of the capacity serving that leg over the
window, at the customer's actual contracted rate, including idle. A placement
that runs at low utilisation is more expensive and the number must say so.

**3.2 Served tokens** counts only output tokens from requests that met both
the latency bound and the quality floor. A late token is not a cheap token, it
is not a token. A retried request carries its full spend and only the final
compliant output.

**3.3 The measured saving** is the difference in C multiplied by the total
compliant work delivered across both legs, minus the holdout cost from 2.1.

**3.4 Both legs are evaluated on the same uniform sample for quality.** Quality
is not evaluated only on latency-compliant requests and multiplied through;
that introduces a conditioning error. Joint sampling gives the joint
compliance rate directly.

---

## 4. Stationarity, and when the period does not count

A comparison across two legs is only meaningful if the traffic reaching them
is the same traffic.

**4.1 Split-half on offered load.** Divide the window in two and compare
arrival rate. Divergence above 10 percent flags the period for review. Traffic
that grew during the window is not a placement result.

**4.2 Split-half on C, per leg.** Same threshold. A leg whose own cost per
served token moved by more than 10 percent between halves was not in a steady
state.

**4.3 Workload shape, at the median.** Median context and generation length,
compared between legs and between halves. A shift in either invalidates the
comparison, since shape is the dominant cost driver and neither party controls
it.

**4.3b Workload mix, across the distribution.** A median can hold while the
distribution moves underneath it. A class serving two tenants, one at 500
tokens and one at 8,000, has the same median whether the split is 50/50 or
20/80, and the cost per served token is very different.

Compare the empirical distribution of context length between legs, and between
halves within each leg, using a two-sample Kolmogorov-Smirnov test at the 1
percent level. The bar is deliberately loose: this is looking for a mix shift
large enough to move cost, not for statistical significance on a large sample,
and a strict threshold on 100,000 requests would flag every period.

Report the statistic and the decile boundaries of both distributions on the
receipt, not only the verdict. A reader who disagrees with the threshold can
apply their own.

**Failing 4.3b flags rather than voids.** A mix shift explains a difference; it
does not necessarily invalidate one. The parties decide whether to bill the
period, and the decision is recorded with the statistic that prompted it.

**4.4 Index of dispersion on arrivals**, reported across a 1, 10 and 60 second
bin ladder. Burstiness is timescale-dependent and a single-bin figure is not
defensible. This is reported rather than gated, because it explains variance
rather than invalidating it.

**A period failing 4.1 through 4.3 is reported, not billed.** The raw data
publishes with the failure noted. There is no version of this where a failed
period is quietly re-run until it passes.

---

## 5. The receipt

Every measurement period produces one record containing:

- The declaration from section 1, verbatim, as signed
- Both legs' C, with a Wilson interval on the compliance proportion
- The measured saving, with the holdout cost shown as a separate line
- The evaluated quality fraction and its sampling error
- The stationarity checks from section 4, with values, not just verdicts
- The serving stack version and configuration for both legs
- A pointer to the raw traces

**The interval on compliance inverts when propagated to cost.** The upper
bound on compliance produces the lower bound on cost per served token. Bill on
the conservative end.

The receipt is a conforming record under the CUW-SLO specification and carries
the same provenance labels. Any figure that was simulated, fitted, or taken
from a configuration file rather than measured is labelled as such on the
face of the document.

---

## 6. Billing

**6.1** The share is applied to the measured saving from 3.3, after the
holdout cost is deducted, at the rate stated in the contract.

**6.2** A period that fails section 4 produces no invoice. It produces a
receipt and a published note.

**6.3 Negative periods carry forward.** A period where the measured saving is
negative produces no invoice, and the loss offsets the next period's saving
before the share is applied. The placement reverts to baseline at the next
boundary unless the customer directs otherwise.

Without carry-forward the arrangement captures a share of every gain and none
of any loss, which is an asymmetry a counterparty is right to refuse. Carrying
the loss forward makes it symmetric, and it costs nothing in the case where
the recommendation was correct.

**6.4 Circuit breaker.** Period boundaries are up to thirty days apart and a
placement that breaches the service level needs reverting in minutes, not at
the next boundary.

The declaration states a breach margin and a window: by default, treatment-leg
p99 exceeding the declared bound by more than 20 percent over any 15-minute
interval with at least 200 requests. On breach the customer reverts
immediately, the period is **voided rather than counted as a loss**, and the
receipt records the breach with its traces. A voided period does not carry
forward under 6.3, because a placement that never held the bound was never a
measurement.

**6.5 Conversion.** Savings-share decays as the easy savings are taken. The
contract states, at signature, the trigger at which billing converts to a flat
platform fee on routed volume: by default, the second consecutive period in
which measured saving falls below half the first period's. Stating it at
signature rather than negotiating at renewal is the difference between a
pricing model and an argument.

---

## 7. Scope and dispute resolution

The questions a counterparty asks before signing, answered in advance so they
do not have to be argued later.

**What a period establishes.** That two placements, under matched traffic,
delivered compliant work at different cost. It is a measurement of difference,
not an attribution to a cause, and it is independent of whether the estimator
predicted it correctly. A period is valid even if the model was wrong about
why.

**Who declares what.** The customer declares the baseline, the service level,
the quality floor and the scope. Reckon measures. Neither party may change the
other's declaration mid-period, and section 1 voids the period if the customer
changes theirs.

**How a disagreement resolves.** The raw traces from both legs are available
to the customer throughout, not on request at the end. Recomputation is
deterministic from those traces using the published method, so a dispute is
resolved by rerunning arithmetic rather than by argument. Where a customer's
recomputation differs from the receipt, the customer's figure governs and the
period is republished.

**The constraints that make the number trustworthy**, all enforced by the
protocol rather than by good intentions:

- The baseline is declared and countersigned before any traffic moves
- The assignment seed is published in advance, so the sample cannot be chosen
- The holdout cost appears as a line item, not netted silently
- Stationarity checks are computed on every period and their values published
- A failed period produces a receipt and no invoice
- All raw traces are the customer's, continuously

**Where savings-share is the wrong instrument.** A workload whose shape is
genuinely non-stationary cannot be measured this way, and section 4.3 detects
that rather than correcting for it. Those customers belong on flat-fee
assurance, and the threshold is arithmetic rather than judgement: see 8.3.

---

## 8. Resolved cases

### 8.1 Multi-tenant attribution

When one placement serves several workload classes, shared capacity cost is
apportioned by **resource-weighted served tokens**, not by request count and
not evenly.

    share_i  =  ( tokens_i x bytes_per_token_i )
                -------------------------------------
                sum over j of ( tokens_j x bytes_per_token_j )

`bytes_per_token` is the decode byte cost from the model's own signature:
active weights plus the class's mean context times the key-value cost per
token. It is computable from the declaration in section 1 without any
measurement, which is what makes it agreeable in advance.

This is proportional to what each class actually consumes from the shared
device. A long-context class pays more per token than a short-context class on
the same hardware, which is correct and is the reason request-count
attribution understates retrieval workloads by a large factor.

**Rule.** Each workload class carries its own contract, its own baseline and
its own holdout, and shares capacity cost by the formula above. Attribution is
recomputed each period from that period's measured token counts.

### 8.2 Baseline refresh over long horizons

A baseline set in month one is stale by month six, because prices move and
model versions ship. Billing forever against a stale baseline would charge a
customer indefinitely for a one-time improvement.

**Rule: the baseline ratchets at each period boundary.**

    baseline(n+1)  =  the cheaper of
                        baseline(n)
                        the placement actually served in period n

The saving in each period is measured against the best placement previously
known, never against the original. Three consequences, all intended:

- A one-time gain is billed once, not annually
- A placement that degrades is caught, because the ratchet does not move up
- Continuity is preserved, since the contract does not void when the baseline
  moves; only the reference point advances

**Non-routine adjustments.** A change the customer initiates that is outside
placement, such as a model swap, a traffic-volume change beyond the
stationarity band, or a new service level, resets the baseline to the new
configuration and starts a fresh period. That reset is a declaration under
section 1, countersigned, and the prior periods stand as measured.

This is the energy performance contracting rule, and it is used because a
ratcheting baseline is the only version both parties will sign twice.

### 8.3 The scale floor

Savings-share has a fixed cost the customer bears: the holdout runs on the
worse placement. Below a threshold that cost exceeds what the arrangement is
worth, and flat-fee assurance is the correct instrument.

With a 5 percent holdout, a 25 percent available premium, and a 20 percent
share:

| Annual spend, per workload class | Net saving | Reckon share | Instrument |
| --- | --- | --- | --- |
| $100,000 | $22,500 | $4,500 | flat fee |
| $250,000 | $56,250 | $11,250 | flat fee |
| $500,000 | $112,500 | $22,500 | savings-share |
| $1,000,000 | $225,000 | $45,000 | savings-share |
| $10,000,000 | $2,250,000 | $450,000 | savings-share |

**Break-even is $333,000 of annual spend per workload class**, against a
$15,000 flat fee. The general form:

    S*  =  flat_fee / ( share x premium x (1 - 2h) )

**Rule.** Below S*, flat-fee assurance. Above it, savings-share, with the
customer keeping 80 percent of the net and paying 1.25 percent of spend for
the holdout. The threshold is computed from the customer's own declared spend
at contracting, not negotiated.

A customer crossing S* mid-contract converts at the next period boundary, and
the conversion trigger is stated at signature.

---

## 9. A worked period

Illustrative figures throughout, derived from a single consistent starting
point so the arithmetic closes. No part of this is a measurement.

### Declaration, countersigned 1 March

| Field | Value |
| --- | --- |
| Baseline placement | 4x H100 PCIe, vLLM 0.9.1, bf16, `--max-num-seqs 64` |
| Service level | p99 first-token latency under 800 ms |
| Quality floor | response parses as valid JSON against declared schema |
| Workload class | `voice-agent-prod`, Llama-3-8B, identified by route prefix |
| Class spend, declared | $480,000 per year, $40,000 per month |
| Instrument | savings-share at 20 percent, spend above S* of $333,000 |
| Holdout fraction | 5 percent |
| Assignment seed | published 28 February, seven days before the period |
| Circuit breaker | p99 above 960 ms over any 15 minutes, 200+ requests |

Recommended treatment: 6x L40S, same stack and configuration. Estimated cost
per served token 1.54x lower, confidence band plus or minus 10.6 percent,
resting on two measured cells and one prior.

### Warm-up, 1 to 2 March

Treatment leg stabilised within 5 percent after 3,140 requests. Those excluded
from both legs, count recorded on the receipt.

### Period, 2 to 31 March

Roughly 123 million requests, about 47 per second, at 120 compliant output
tokens each.

| | Baseline leg | Treatment leg |
| --- | --- | --- |
| Share of traffic | 5 percent | 95 percent |
| Spend over the window | $2,000 | $38,000 |
| Compliance rate | 93.8 percent | 95.6 percent |
| Compliant served tokens | 485 MSVT | 14,232 MSVT |
| **Cost per million served tokens** | **$4.12** | **$2.67** |

Quality evaluated on a 2 percent uniform sample of both legs, assignment
decided after the response was produced. The evaluated fraction and its
sampling error carry into the interval.

### Stationarity

| Check | Value | Verdict |
| --- | --- | --- |
| 4.1 offered load, split-half | 3.1 percent divergence | pass |
| 4.2 cost per served token, split-half | baseline 2.8, treatment 4.1 percent | pass |
| 4.3 shape, medians | context 1,466 vs 1,471; generation 118 vs 119 | pass |
| 4.3b mix, KS on context | D = 0.011, p = 0.34 | pass |
| 4.4 index of dispersion | 2.1 at 1s, 4.7 at 10s, 9.2 at 60s | reported |
| 2.7 assignment balance | 4.98 percent realised against 5.00 declared | pass |

### Settlement

    Compliant work, both legs                14,718 MSVT
    Baseline cost per MSVT                        $4.12
    Treatment cost per MSVT                       $2.67
    Difference                                    $1.45

    Gross saving       14,718 x $1.45        =  $21,341
    Holdout cost          485 x $1.45        =     $704
    Net saving                               =  $20,637

    Reckon share, 20 percent                 =   $4,127
    Customer retains                         =  $16,509

    Carry-forward from prior period                   0

Wilson interval at 95 percent on compliance: baseline 93.6 to 94.0 percent,
treatment 95.5 to 95.7. Propagated to cost, the upper bound on compliance
gives the lower bound on cost per served token, so the invoice issues at the
conservative end: **$4,050**.

Annualised, the customer retains roughly $198,000 against a $49,500 fee on a
$480,000 class.

### What the receipt carries

The declaration verbatim, both legs' figures, all six checks with their
values, the excluded warm-up count, the holdout cost on its own line, the
interval, the serving-stack versions for both legs, and a pointer to the trace
records the customer already holds.

### The same period, had the recommendation been wrong

Treatment measured at $4.61 per MSVT against the baseline's $4.12. Net saving
of negative $7,100.

No invoice. The placement reverts at the boundary. The loss carries forward
against the next period's saving under 6.3, so the following period bills only
on the amount by which it exceeds $7,100. The receipt publishes with the
traces, and the failed recommendation appears in the corpus alongside the
successful ones.

---

*Published under CC BY 4.0. Adopt it, fork it, or argue with it. A method only
one party may use is not a method.*
