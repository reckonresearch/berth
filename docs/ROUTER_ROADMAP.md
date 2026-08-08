# The router: what to build

**Adaptive placement.** A control plane that decides where a workload runs,
moves it when the answer changes, and proves what the change saved. It never
touches a request.

---

## MVP: three components, all buildable now

### 1. Decision

`berth place` takes a workload signature and a service level and emits a
ranked placement with a confidence band, plus a feasibility verdict.

Mostly exists. `berth estimate` produces the ranking, `kv_pressure` produces
the feasibility verdict, and the Index carries the corpus behind it. What is
missing is the wrapper that turns an estimate into a **decision record**: the
recommended placement, the alternative it beat, the margin between them, and
whether that margin exceeds the confidence band. A recommendation inside the
noise floor is not a recommendation.

### 2. Holdout

Given a declared baseline, emit an assignment function: a hash of request
identifier against a pre-committed seed, returning baseline or treatment.
Plus the period definition, the stopping rule, and the stationarity gates.

Specified in full in the Placement Holdout Protocol. Not built. It is roughly
two hundred lines and it is the piece that makes a savings claim contractible.

**The customer runs the assignment.** We never see a request, which is what
keeps this a control plane.

**And it must not require application code.** Asking a team to ship a vendor's
function into their production request path, through their review and deploy
process, for a company they are still evaluating, is a far higher barrier than
installing a CLI. Most integrations die there.

So the assignment ships as a **deployment configuration change**, not a code
change:

- Two endpoints registered, baseline and treatment
- Weighted routing at the gateway or load balancer already in front of them,
  set to the declared fraction
- Assignment by the gateway's own consistent hashing, seeded with our
  published value

Envoy, NGINX, Istio, ALB and every managed gateway support this natively. The
customer changes a config file. Nothing of ours runs in the request path, and
the assignment is auditable by them without reading our code.

**Fallback for teams without a gateway:** a reference implementation of the
hash in Python, Go and TypeScript, published with the protocol. Used by
choice, not by requirement.

### 3. Verification

Ingest traces from both legs, compute cost per served token for each, produce
the delta with a Wilson interval, run the four stationarity checks, and emit a
conforming receipt.

Partly built. `sounding` records traces, the specification defines the record,
`bench.holdout` already computes intervals across splits. What is missing is
the two-leg comparison and the receipt writer.

**Total new code for the MVP: roughly 1,500 lines, two to three weeks.** The
receipt writer alone carries the Wilson intervals, the four stationarity
checks and conforming-record output, and that is most of it. The earlier
estimate of six hundred lines counted the happy path and not the arithmetic.

It is not blocked on engineering. It is blocked on one customer willing to
declare a baseline.

---

## Beyond MVP

Ordered by what each unlocks, not by difficulty.

### Tier 1: makes the position hold

**The placement agent, delivered as a pull request.**

This is the feature that converts assurance from a quarterly report into a
service the customer notices working, and it is first because it is what
renews the contract.

The loop:

1. **Watch.** Model release feeds for every model in a customer's declared
   classes: Hugging Face, vendor registries, and the serving stacks' own
   release notes.
2. **Detect.** A version change in a declared stack, a provider price change,
   or a new corpus cell touching that class.
3. **Re-estimate.** Run the placement decision against the new inputs. If the
   optimal placement is unchanged, or the margin sits inside the confidence
   band, stop silently. Most triggers should end here, and an agent that opens
   a pull request every week is an agent that gets muted.
4. **Open a pull request** against the customer's deployment configuration, in
   their repository, containing:
   - the config diff, the actual change, not a recommendation to make one
   - the estimate before and after, with confidence bands
   - which corpus cells the new answer rests on and whether they are measured
     or prior
   - the expected saving, and the holdout design if the class is on
     savings-share
   - a link to the raw traces

**Why a pull request rather than an alert.** An alert creates work. A diff with
evidence attached is work already done, arriving in a review workflow every
infrastructure team already has, where it can be discussed, amended, rejected,
or merged. It also produces a record of the decision inside the customer's own
history rather than inside a vendor dashboard.

**What it never does.** Merge. Deploy. Touch anything outside the declared
configuration paths. The agent proposes and the customer disposes, and that
boundary is in the contract rather than in the implementation.

**What it requires.** Read access to a configuration repository and write
access to a branch. Nothing else, and specifically not access to traffic,
credentials, or the request path.

**Model-release triggers.** Watch the registries a customer declares. When a
model version they run ships an update, re-estimate automatically and alert if
the optimal placement changed. This is the frequency argument made
operational, and it is what converts assurance from a quarterly report into a
service that notices before the customer does.

**Price-change triggers.** Provider rate cards move. A placement optimal at
$2.60 an hour may not be at $1.90. Poll and re-estimate.

**Drift detection on the live leg.** Compare current cost per served token
against the period's opening value. A placement degrading mid-period, from a
noisy neighbour, a driver change, or a silent provider migration, should
surface before the period closes rather than after.

**Corpus-move re-estimation.** When a new cell lands that touches a customer's
workload class, their recommendation is recomputed. A prior becomes a
measurement and the confidence band narrows. This is the flywheel made visible
to the customer paying for it.

**Revert path.** Any placement the agent proposed can be reverted by reverting
the commit. That is worth stating explicitly, because it is the answer to the
first objection an infrastructure lead raises, and it is true only because the
agent works through configuration rather than through an API.

### Tier 2: makes it multi-workload

**Class-level portfolio placement.** An organisation running twenty workload
classes has twenty placements interacting through shared capacity. Solve them
jointly rather than one at a time. This is where Insull's diversity factor
becomes a product feature: mixing a tight-deadline class with a loose one on
the same capital raises the load factor, and neither class in isolation shows
it.

**Deadline-class mixing.** Explicitly schedule loose-SLO work into the
capacity headroom that tight-SLO work requires but does not use. This is the
PENDING column of the 39.5x decomposition, and it is a routing decision rather
than a placement one.

**Feasibility envelopes as a first-class output.** Not just "this placement
costs X" but "this workload cannot be served on one of these below batch 24."
The estimator already knows; the router should say it before a customer buys
the wrong capacity.

### Tier 3: makes it financial

**Forward placement.** Given a growth projection, what capacity should be
committed and at what term. This is what turns a placement engine into an input
for reserved-capacity decisions, which is the highest-value moment in a
customer's year.

**Counterfactual as a service.** Verification for contracts we did not
originate. Two other parties write a delivered-work contract, we verify the
period. Fee business, volume independent of our headcount.

**Portfolio-level exposure.** For an allocator or a lender: given a fleet and
a set of workloads, what is the productivity of the fleet, and how does it
move if the software deficit closes at the observed rate. This is the corpus
data product with a model attached.

### Tier 4: the boundary

**Learned surrogates where the closed form breaks.** Disaggregated prefill and
decode, chunked prefill, speculative decoding, multi-node interconnect. At
some point arithmetic stops being the right object.

The closed form's boundary is itself the contribution. A surrogate trained
without knowing where the physics holds is a black box; trained with it, you
know which regions need learning and which are simply arithmetic. That
distinction is worth more than either component alone, and it is only
available to whoever mapped the boundary first.

---

## What we will not build

**Anything in the data path.** No proxy carrying production traffic, no
gateway, no failover, no SRE surface. The moment we carry a request we have an
availability obligation, and the business becomes infrastructure rather than
measurement.

**Capacity.** No resale, no reserved blocks, no inventory. Inventory creates a
reason to route toward it.

**Model quality evaluation.** We define the interface for declaring an
acceptance test and the rules for sampling it. We do not define what quality
means for a customer's workload. That is a crowded, contested, different
company.

---

## What each track is blocked on

The sequence below reads as one line of work. It is three tracks, gated
differently, and treating them as sequential is the fastest way to stall.

| Track | Blocked on | Can start |
| --- | --- | --- |
| Decision record | nothing | now |
| Holdout assignment, reference implementations | nothing | now |
| Gateway config templates | nothing | now |
| Verification and receipt writer | nothing | now |
| **The placement agent** | **nothing**, it runs against public registries and a config repo | **now** |
| First live period | one customer declaring a baseline | on signature |
| Model-release triggers, in production | first period closed | after |
| Corpus-move re-estimation | more cells than we have | ~15 cells |
| Class-level portfolio placement | multi-class customer | second account |
| Deadline-class mixing | the pre-registered mixing run | that cell |
| Forward placement | premium time series, several quarters | 2027 |
| Learned surrogates | the cofounder | that hire |

**Five of thirteen items are unblocked today.** Two are blocked on a customer,
three on measurement, one on a hire, and two on time.

**The agent is unblocked and it is the highest-value unblocked item.** It can
run against our own corpus for a quarter before any customer depends on it,
which is also how it earns the right to be trusted with a pull request.

## Capacity

Ten to twelve weeks of engineering, sequenced for one person who is also doing
outreach, measurement, fundraising and a cofounder search. That is not a
schedule, and pretending it is would make every date here fiction.

**With one person:** items 1, 2, 3 and 5 are achievable in six weeks at
perhaps half time. Item 4, the receipt writer, waits for a customer, since
building it against an imagined declaration is how it gets built wrong.

**With the cofounder:** the same list is three weeks and Tier 1 follows
immediately.

**What gets cut if neither happens:** Tier 2 and Tier 3 entirely, and the
roadmap becomes decision record, holdout, agent, and one live period. That is
still a product and it is still the whole thesis. Everything past it is
expansion.

## Kill criteria

Stated per tier, because a roadmap with no stopping condition is a wish list.

**The agent.** If, over its first two quarters running against our own corpus,
fewer than one trigger in four produces a placement change that clears the
confidence band, the trigger set is wrong and the feature is noise. Retune or
drop it.

**Class-level portfolio placement.** If solving classes jointly does not beat
solving them independently by more than the confidence band on a real
multi-class account, it does not ship. Joint optimisation that ties is
complexity with a story attached.

**Deadline-class mixing.** If the pre-registered mixing run shows less than a
1.2x improvement in delivered work per unit capital, the PENDING column of the
39.5x decomposition is smaller than claimed and both the roadmap and the deck
change to say so.

**Forward placement.** If the premium time series shows the spread between
placements narrowing faster than new silicon widens it, the decision has a
shelf life and forward commitment advice is not a product.

**Learned surrogates.** If the closed form holds within gate across
disaggregated serving and speculative decoding, there is no boundary to map
and this tier does not exist. That would be good news and it should be
published.

## Sequence

1. **Decision record.** One week. Turns an estimate into something a contract
   can reference: recommended placement, the alternative it beat, the margin,
   and whether that margin clears the confidence band.
2. **Holdout assignment and period logic.** One week, plus the reference
   implementations in three languages. Published alongside the protocol,
   which is how it gets adopted by people who are not customers.
3. **Gateway configuration templates.** Two days. Envoy, NGINX and ALB,
   because the integration barrier is the thing most likely to kill a first
   deployment and a working config removes it.
4. **Two-leg verification and the receipt writer.** Two weeks.
5. **The placement agent.** Two weeks, and it can be built before the first
   customer because it operates on a public model registry and a config
   repository rather than on traffic. Ship it against our own corpus first, so
   it has run for a quarter before anyone depends on it.
6. **First live period with a design partner.** Gated on a customer, not on
   code.

Everything in Tier 1 is a month of work behind the MVP. Tiers 2 and 3 are the
Series A. Tier 4 is the research programme and the reason the cofounder seat
exists.
