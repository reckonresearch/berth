# Reckon Research: state of the build

**Checkpoint date: 8 August 2026.** Re-run this audit whenever a decision feels
like it rests on something nobody has checked recently.

This is the single source of truth. Where it disagrees with a slide, a post or
a memory, this document is wrong and should be corrected, or the other thing
is and should be. Either way the disagreement is the finding.

---

## 1. What the company is

**Category: adaptive placement.**

Ornn tells you what compute costs. OpenRelay sells you compute cheaply. Reckon
tells you where your work should run, moves it there, and moves it again every
time the answer changes.

**The gap.** Compute is bought by the GPU-hour and consumed as work delivered
under a deadline. No standard unit bridges the two, so there is nothing to be
efficient per. Tokens per second can double while a service delivers nothing
sellable, because a token arriving after its deadline is not a token.

**The unit.** One output token delivered inside its latency target and above a
declared quality floor. A served token.

**The frequency, which is what makes it a business rather than an audit.**
Every model version silently invalidates a placement, and nobody checks. That
happens monthly or faster and it is visible on a release calendar.

---

## 2. Products, and what each is named

| Name | What it is | Status |
| --- | --- | --- |
| **berth** | the estimator. Closed-form roofline plus a queueing term, predicting cost and latency on a given accelerator before it is rented | shipped, Apache-2.0, on PyPI as `berth-placement` |
| **sounding** | the harness. Drives a live endpoint and produces the traces that check the estimator. Never imports the estimator, by design and by test | shipped |
| **The Placement Index** | the public reference. Ranks placements rather than models, workload-conditional | live at reckonresearch.com/index/ |
| **Pilot** | the control plane. Decides where a workload runs, watches for change, and proposes a move as a pull request. Never touches a request | built, untested against a customer |
| **The declaration** | `.berth/classes.yaml` in the customer's own repository. What Pilot may watch, which paths it may write, what bound each class holds | built |
| **The status page** | `.berth/STATUS.md`, rewritten on every pass. Every class, what it holds, what is recommended now, open proposals, settled periods, sources polled | built |
| **Receipts** | the artifact. A conforming record settling one measurement period | built |
| **versus** | self-host or API, on one axis. Cost per compliant token both sides, with the break-even volume | built |
| **CUW-SLO** | the specification. How a conforming measurement is made and recorded | written |
| **The Placement Holdout Protocol** | how a saving is proven. Declared baseline, held-out traffic, verified delta | written, never run |

**On the name Pilot.** A harbour pilot boards a vessel, guides it to berth,
owns neither the ship nor the port, and is paid by the vessel. That is exactly
the position: we route on behalf of the buyer, to any provider, and no
provider ever pays us. The name is doing work rather than decorating.

**Not yet used publicly.** The deck and the site say "the router" and
"adaptive placement". Pilot should replace "the router" everywhere, because a
router is a thing OpenRelay sells and this is not that.

---

## 3. What is built

### berth, the estimator

`berth/` at 178 passing tests, lint clean, CI green.

- `estimate.py` roofline, queueing, feasibility, KV pressure
- `place.py` the decision record a contract references
- `holdout.py` the protocol: assignment, declaration, period, stationarity
- `receipt.py` two-leg settlement and the conforming record
- `agent.py` watch, detect, propose, and the state that stops it repeating
- `watch.py` registry, price and corpus watchers
- `github.py` opens the pull request and refuses everything else
- `quantities.py` typed ceilings that make two unit defects impossible
- `workload.py`, `silicon.py`, `queueing.py`, `calibrate.py`, `classes.py`

CLI: `estimate`, `premium`, `place`, `pilot`, `holdout`, `list`.

### sounding, the harness

`bench/` with the sweep, microbenchmark, validator, contamination auditor,
cross-validation, and `p0_run.sh` which captures silicon identity, serving
stack, launch flags and quantization, and refuses when a declared label
contradicts the server.

### The corpus

Six measured cells, three memory technologies, two vendors, two serving
stacks.

| Silicon | Model | Precision | Stack | Note |
| --- | --- | --- | --- | --- |
| L40S | Llama-3-8B | bf16 | vLLM 0.6.3 | validated |
| L40S | Llama-3-8B | bf16 | SGLang | stack transfer |
| H100 PCIe | Llama-3-8B | bf16 | vLLM 0.6.3 | validated |
| H100 SXM | Llama-3-8B | bf16 | vLLM 0.5.5 | version confound, re-running |
| H100 SXM | Llama-3-8B | fp8 both | vLLM 0.25 | fully captured |
| A100-80G | Qwen3-30B-A3B | bf16 | vLLM 0.26 | MoE, gate not met |
| MI300X | Llama-3-8B | bf16 | vLLM 0.25 | first AMD |

### Published results

**Leave-one-silicon-out: 9.5 percent.** Fit the decode constant on one card,
predict a card the model has never seen, against a 15 percent gate published
before the first run. The residual is the true spread between the cards, 0.825
and 0.761, and one constant across both costs about nine percent. That is the
price of generality, stated rather than fitted away.

**Serial prefill transfers across schedulers.** Effective parallelism 1.09
under vLLM and 1.01 under SGLang. Modelling the term drops first-token error
from roughly 65 percent to under 9.

**Decode is stack-independent to three decimals.** 0.854 under SGLang against
0.850 under vLLM, same card, same model.

**MI300X reads paged key-value cache well at low batch and badly above 8.**
3,854 GB/s at batch 4 falling to 633 at batch 16, while both NVIDIA cards hold
flat. At batch 1 it delivers 92 percent of its own microbenchmark, so the
memory system is fine and the kernel is not. End to end the card is twice as
fast as an H100 SXM at low batch and 1.4 times slower at batch 32 with long
prompts.

**That last one is the most valuable measurement in the corpus.** It is the
first cross-vendor finding, it separates silicon deficit from software deficit,
and it is the workload-conditional premium the deck claims and could not
support until now.

### Instrument defects

Ten found, all published in `DEFECTS.md`, each pinned by a regression test.

**Four let bad data through. Six rejected good data.** The asymmetry is the
point: a false negative is caught downstream eventually, a false positive is
silent, and a design partner whose clean file is called contaminated does not
debug the tool, they stop using it.

Only physical impossibility now refuses a file. Everything else prints and
passes.

---

## 4. Business model

### Revenue

| Stream | Price | Status |
| --- | --- | --- |
| berth and the Index | free | live |
| Placement assurance | $7,500 to $15,000 per workload class per year | no customer |
| Routing, above the floor | 20 percent of measured savings | no customer |
| Corpus access | $60,000 per seat | no customer |
| Implementation | $25,000 first class | no customer |
| Verification | fee, unpriced | needs the spec adopted |

**The pricing unit is one workload class: one model, one serving
configuration, one service level.** Change any and it is a new class, which is
exactly where the answer changes, and it is verifiable from a deployment
config rather than from a conversation.

**The scale floor is arithmetic.** `S* = flat_fee / (share x premium x (1-2h))`,
which is $333,000 of annual class spend at the defaults. Below it the holdout
costs more than the arrangement returns.

### Market, counted from buyers

1,710 organisations self-hosting at material scale, $69M serviceable, $10M at
15 percent share by year five. Routing is separate and conditional: 3 percent
of inference routed at a 2 percent platform fee is $153M, and it is worth zero
unless savings-share converts to a platform fee on the first three contracts.

### What is defensible

Not the instrument, which is open source. Not the corpus, which anyone can
rebuild for four dollars a cell. Not the index, which can be forked.

**The counterfactual.** Being the party whose declared baseline both sides
accept is what converts a measurement into a billing event, and leaving means
renegotiating every baseline.

---

## 5. What is missing

### The buyer set, and the lever on it

The market slide counts 1,710 organisations self-hosting at material scale.
That number is the constraint, and everything before `versus` assumed the
decision to self-host was already made.

`berth versus` answers the question most teams actually have. What a hosted
endpoint charges is published; what the same work would cost on hardware they
would rent is not, and computing it is the one thing this instrument can do
that nothing else can. It puts both on one axis, excludes any offer whose
latency misses the bound, and names the volume where the answer flips.

The early runs make the shape clear. At 1,469 prompt tokens under an 800 ms
bound, one MI300X saturates at roughly 40,700 requests an hour. Below about
33,000 an hour a rented node is paid for while idle and the API wins at any
price. Above it self-hosting wins, but only if engineering time is near zero:
at twelve dollars an hour of loaded operator cost, self-hosting does not win at
any volume.

**That last figure is the finding.** For most teams the deciding term is not
the GPU, it is the person watching it, and no price sheet on either side says
so.

### Blocking, and not on us

- **A design partner.** Every paid product needs one and none exists. The
  holdout protocol has never been run, the receipt has never settled a real
  period, and Pilot has never opened a pull request in anyone's repository.
- **A cofounder.** Ten instrument defects in one week, none caught by review.
  That is the argument, from evidence.

### Blocking, and on us

- **Docs for eight modules.** `place`, `holdout`, `receipt`, `agent`, `watch`,
  `github`, `declaration` and `quantities` have no page on
  docs.reckonresearch.com. The code is public and undocumented, which is worse
  than either alone.
- **The protocol, business model and roadmap live in a chat, not the repo.**
  `HOLDOUT_PROTOCOL.md` is the asset that makes savings-share contractible and
  it is not published.
- **Pilot is not named publicly.** The deck says "the router".

### Cheap and never done

- Search Console and Bing sitemaps. Ten minutes, multi-week clock, gates
  nothing.
- PyPI token rotation to project scope.
- Three pre-registrations published to `/writing/`, which is empty.
- Three Substack posts written and unpublished.
- Two deck screenshots: `berth estimate` output and the Index.

### In flight

- Cells A and B on H100 SXM, closing the version confound
- The denser MI300X batch ladder, to locate the knee
- Outreach to five targets that own their weights, plus Hebbia

---

## 6. Technical roadmap, and what each is blocked on

| Item | Blocked on | Can start |
| --- | --- | --- |
| Decision record | nothing | **built** |
| Holdout assignment and period | nothing | **built** |
| Gateway config templates | nothing | **built** |
| Verification and receipt | nothing | **built** |
| The agent | nothing | **built** |
| Registry and price watchers | nothing | **built** |
| GitHub integration | nothing | **built** |
| First live period | one customer declaring a baseline | on signature |
| Model-release triggers in production | first period closed | after |
| Corpus-move re-estimation | more cells | ~15 cells |
| Class-level portfolio placement | a multi-class customer | second account |
| Deadline-class mixing | the pre-registered mixing run | that cell |
| Forward placement | premium time series | 2027 |
| Learned surrogates | the cofounder | that hire |

**Seven of fourteen are built.** Two are blocked on a customer, three on
measurement, one on a hire, one on time.

### Kill criteria

- **The agent.** Fewer than one trigger in four producing a change that clears
  the confidence band means the trigger set is wrong and it is noise.
- **Class-level portfolio placement.** If solving jointly does not beat solving
  independently by more than the band, it does not ship.
- **Deadline-class mixing.** Below 1.2x improvement in delivered work per unit
  capital, the PENDING share of the 39.5x is smaller than claimed and the deck
  changes.
- **Learned surrogates.** If the closed form holds across disaggregated serving
  and speculative decoding, there is no boundary to map. That is good news and
  publishes.

---

## 7. How the two rounds align

**The seed buys the answer to one question:** does a buyer pay a share of
measured savings, or only a flat fee. If the first, this is a company at the
scale the market slide describes. If the second, it is a very good $30M tools
business, and eighteen months is enough to know.

| Seed delivers | Series A requires |
| --- | --- |
| 40 cells across NVIDIA, AMD, Google TPU, AWS Trainium | 3 design partners running berth in production |
| Dense, MoE and MLA, bf16 and fp8 | trace-return rate above 30 percent, published monthly |
| 3,600 traces, a third returned by partners | first paid receipts contract signed |
| Pilot live: the agent proposes, the customer merges | the Index cited by someone who did not run it |

**$1.5M, 24 months, conditional on the cofounder.** Raising against two open
seats means the first question is what happens if the hire does not land, and
there is no good answer today. If the seat is still open in six weeks, $750k
and twelve months is the honest version.

**The free test, available in any booked call:** *if we could prove you saved
$400k, what would you pay for that proof?* A percentage-shaped answer is the
$1B path. A flat few thousand is the tools business. It costs nothing and it
resolves more than any model.

---

## 7b. What is public and what is not

Nothing already published is pulled back; that costs more credibility than it
protects. The line going forward is the one performance contracting drew:
**the protocol is public and the measurements are the product.** IPMVP is
public. An energy service company's data on a specific building is not.

**Public, permanently**

| | Why |
| --- | --- |
| The unit specification | A unit nobody may use is not a unit |
| The Placement Holdout Protocol | Adoption by non-customers is the whole verification revenue line |
| berth, the estimator | Anyone can re-run our numbers is the entire credibility claim |
| sounding | Contributors need it |
| The receipt format | A record only we can read is not a record |
| Gateway templates and the declaration schema | Removes the integration barrier |
| DEFECTS.md | Nobody else publishes this |
| The agent's suppression logic | The claim that we do not spam you has to be checkable |
| The status page format | It lives in the customer's repo; a format only we can render would defeat that |

**Private from here**

- **Calibration constants beyond published cells.** The model is open and the
  fitted values are the product. `place.py` now loads bands from data via
  `BERTH_CORPUS_BANDS`, so the published corpus and a licensed one are
  different files against the same estimator.
- **The tuned trigger set.** Publishing how to poll a registry is fine.
  Which changes actually move a placement, at what thresholds and on what
  cadence, is learned by running it and belongs in configuration.
- **Everything customer-specific.** Baselines, receipts, traces, declarations.
- **The hosted operation.** Scheduling, state, credentials.

---

## 8. Standing rules

- **Provenance on every number.** MEASURED, FITTED, CONFIG, SIM, HYPOTHESIS.
  A number labelled MEASURED with a simulated input is the one unrecoverable
  error.
- **Never publish a figure a run has not produced.** The whole claim is that
  the figures are checkable.
- **Subtract the floor before taking any ratio from first-token latency.** Six
  occurrences, including inside documentation warning against it.
- **If the data carries a fact, no check may assume it.**
- **A guard needs a test for what it permits, not only what it blocks.** Six of
  ten defects rejected correct data and every one had only the failure path
  tested.
- **Only physical impossibility refuses a file.**
- **No provider ever pays.** Not for routing, not for ranking, not for
  placement in the Index.
- **Never own capacity.** Inventory creates a reason to route toward it.
- **The instrument never imports the estimator.** Evidence that cannot be
  separated from the thing it evaluates is not evidence.
- **Pre-register before measuring.** Scope declared afterwards is selection.
- **Publish the failures.** A measurement tool that has never been wrong in
  public has not been used.

---

## 9. The two numbers

Everything above is infrastructure. Two numbers decide whether this is a
company, and neither has moved:

**Traces returned by someone who is not us: zero.**

**People on the team: one.**

Both are one conversation away.
