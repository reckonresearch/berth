# Reckon Research: business model

**Category: adaptive placement.**

Ornn tells you what compute costs. OpenRelay sells you compute cheaply. Reckon
tells you where your work should run, moves it there, and moves it again every
time the answer changes.

---

## The problem, stated at the frequency it actually occurs

Cost anxiety is a slow leak and nobody buys against it. The event that
triggers a purchase is specific and it repeats:

**Every model version silently invalidates your placement, and nobody checks.**

A team ships Llama 3.1, then 3.3, then Qwen 3. Same hardware, same
configuration, same reserved capacity. The byte profile changed, the attention
shape changed, the optimal placement changed, and the deployment did not. That
happens monthly or faster, it is visible on a release calendar, and it is
worth between 20 and 60 percent of the inference bill each time.

Four other events, each of which surfaces the same gap:

- A reserved-capacity commitment comes up for renewal
- p99 breaks and the cause is not in the application
- The inference bill grows faster than traffic
- New silicon becomes available and nobody can say whether it is better

**Placement is a position that decays, not a decision that is made once.** That
is the entire commercial premise.

---

## Revenue architecture

Five streams. Two free by design, three paid, and the last two scale without
consuming founder attention.

### Free: the instrument and the index

**berth**, Apache-2.0, forever. Anyone can reproduce any number, including
against us.

**The Placement Index**, published quarterly, free and citable.

These are acquisition, not charity. A reference nobody reads is worth nothing,
and a measurement nobody can check is worth less. Acquisition cost falls to
publication cost, which is the cheapest funnel available.

### The pricing unit

**A workload class is one (model, serving configuration, service level)
triple.** Change any of the three and it is a new class.

That boundary is not arbitrary: it is exactly where a placement decision
changes. A different model moves the byte profile. A different serving
configuration moves the scheduler behaviour. A different service level moves
which placements are feasible at all. Two workloads sharing all three have the
same answer and should not be billed twice.

It is also verifiable from the customer's own deployment configuration rather
than from a conversation, which is what stops an organisation running twenty
services from declaring one class called "inference."

### Primary: placement assurance

**Priced from declared annual spend on the class, with one threshold formula
governing all three instruments.**

| Class spend per year | Instrument | Price |
| --- | --- | --- |
| Under $250,000 | Assurance, light | $7,500 |
| $250,000 to $333,000 | Assurance | $15,000 |
| Above $333,000 | Savings-share | 20% of net measured saving |

The threshold is arithmetic, not negotiation:

    S*  =  flat_fee / ( share x premium x (1 - 2h) )

At a 20 percent share, a 25 percent available premium and a 5 percent holdout,
S* is $333,000. Below it the holdout costs more than the arrangement returns.
A customer crossing S* converts at the next period boundary, and the trigger
is stated at signature.

**Why tiered rather than flat.** At $100,000 of class spend a $15,000 fee is
15 percent of the bill and nobody pays it. At $1,000,000 it is 1.5 percent and
it is underpriced. The tier exists because the same product is worth different
amounts at different scales, and pretending otherwise loses the bottom of the
market and underprices the top.

Continuous monitoring against a declared service level, re-measurement when
the corpus moves or a model version ships, and a conforming receipt each
quarter. When the answer changes, we open a pull request against the
customer's deployment configuration with the diff and the evidence attached.

**The renewal thesis is model churn, and it is measurable rather than
asserted.** Over the last twelve months, the models in our registry shipped
version updates at a rate that changed the optimal placement in a
material fraction of cases. That figure is computable from the corpus and is
published quarterly alongside the Index, because it is the number that decides
whether assurance renews. A customer whose stack never changes does not need
this product, and we would rather they cancel than be sold it.

Expansion is by workload class, not by seat. An organisation running twenty
distinct services is a twenty-class account, and class count grows with the
customer rather than with our sales effort.

**Why this is the primary rather than the router.** It is billable on day one,
it requires no traffic access, and it is the product that converts a
model-release event into a recurring position. The router is worth more per
account and reaches fewer of them.

### Transaction: routing

**A share of measured savings, converting to a platform fee.**

Governed by the Placement Holdout Protocol. The customer declares a baseline,
a declared fraction of traffic stays on it, and we bill a share of the verified
delta. Above $333,000 of annual spend per workload class this beats the flat
fee for both parties; below it, assurance is the correct instrument and the
threshold is arithmetic rather than negotiation.

Converts to a flat platform fee on routed volume at the second consecutive
period where the measured saving falls below half the first, a trigger stated
at signature.

**No provider ever pays.** Not for routing, not for ranking, not for
placement in the Index. We route on behalf of the buyer, to any provider, and
the buyer is the only party who pays us. That is the fee-only advisor
structure: not neutral about outcomes, aligned with one side, and the
alignment is what makes it credible.

**We never own capacity.** No resale, no reserved blocks, no inventory. The
moment inventory exists there is a reason to route toward it.

### Data: the corpus

**$60,000 per seat per year.**

Four series that require running the same workload on hardware you do not sell,
which is why nobody else produces them:

- **Cost per served token** by silicon, model, serving stack and service level
- **The placement premium** by workload class, which is what capacity is
  actually worth to a buyer
- **Premium time series**, tracking how the answer moves as silicon ships and
  prices change
- **Software deficit closure rate**: how fast non-NVIDIA stacks close on the
  reference stack, per unit time

The last is the highest-value series in the set. Anyone underwriting
non-NVIDIA capacity is making a bet on stack maturity, and there is currently
no number for that rate.

**Buyers, in order of willingness to pay:** capital allocators and lenders,
whose collateral value depends on productivity rather than on the asset price;
silicon vendors buying competitive intelligence on their own parts; neocloud
capacity planners deciding what to buy next.

Purchasers cannot buy what is published, when, or in what order, and every
purchase is disclosed on the Index.

### Implementation

**$25,000, one time, for the first class in an account.**

Declaring a baseline, scoping a class in the customer's own logs, wiring the
assignment at their gateway, and running the first period end to end. Charged
because it takes real engineering time and because a customer who pays for
implementation completes it.

**Design partners pay nothing**, for implementation or for the first year of
assurance, in exchange for the cell being published. That exchange is stated
in the contract rather than left as goodwill, and it is capped at three
partners so it stays a program rather than a discount.

### Financial: verification

**A fee per verified contract, on contracts we did not originate.**

Once delivered-work contracts exist, someone has to confirm they were
performed. Three instruments need a work-denominated basis and have none:

- Settlement for contracts denominated in served tokens under a service level
- Underwriting for inference latency and availability policies
- Savings-share verification between two other parties

Energy service companies do not verify their own savings; a third party does,
under a published protocol. That is a fee business whose volume grows with
other people's contracts rather than with our headcount.

**This is the stream that scales without attention.** It requires the
specification and the holdout protocol to be adopted by parties who are not
our customers, and publishing under CC BY is necessary rather than sufficient.
Adoption happens when using the format is easier than inventing one, so four
things ship alongside it:

- **A conformance suite.** Anyone can run it against their own records and get
  a pass or fail, without contacting us.
- **A reference implementation of the holdout assignment function**, in the
  three languages that cover most admission paths. A protocol whose hardest
  part is a hash function nobody wants to write themselves does not get
  adopted.
- **A public register of conforming records**, listed whether or not we
  produced them. A register that only contains our own work is a portfolio.
- **A published dispute procedure**, so a counterparty can see how a
  disagreement resolves before agreeing to be measured.

---

## Comparison

| | Ornn | OpenRelay | Reckon |
| --- | --- | --- | --- |
| Sells | capacity, data, finance | capacity | decisions |
| Unit | GPU-hour | GPU-hour and token | served token |
| Prices | the asset | the asset | the asset's output |
| Holds inventory | yes | yes | never |
| Provider-funded | yes, supply side | yes, supply side | never |
| Marginal cost | capacity to fill | capacity to fill | roughly $4 per cell |
| Scales with | volume traded | tokens routed | classes held, then verifications |

Ornn prices the input. We price the output. A futures contract on a GPU-hour
cannot settle a hedge on inference cost, because the exposure is delivered
work and the contract references a machine. Whoever owns the work unit owns
that instrument.

Their index is an input to ours, and ours is an input to any instrument
denominated in work.

---

## Unit economics

| | |
| --- | --- |
| First receipt in a new cell | $400, of which $4 is compute |
| Second receipt in the same cell | $5 |
| Marginal cost at corpus depth | under $2 |
| Assurance, per class per year | $15,000 |
| Routing, above the scale floor | 20 percent of net measured saving |
| Corpus access, per seat | $60,000 |
| Verification, per contract period | fee, not yet priced |

**Gross margin improves with corpus depth.** A cell already measured is a
lookup. Price does not fall with depth; cost does. That is the same
compounding as the moat, appearing on the income statement.

---

## Path to scale

**Assurance** carries the first three years. It is billable immediately,
expands by class inside an account, and is triggered by an event that recurs
monthly.

**Routing** carries years three through six. Higher value per account, fewer
accounts, and it requires the holdout protocol to be proven on live traffic
first.

**Corpus and verification** carry the outer years and are the only streams
that grow without proportional effort. Both depend on the specification being
adopted by parties who are not customers, which is why it is published under
CC BY and why no part of it is proprietary.

**The sequence is forced.** Assurance proves the measurement is worth paying
for. Routing proves the counterfactual is defensible. Verification is only
possible once other parties are writing contracts against the unit. Attempting
any of them out of order fails for the same reason: you cannot bill on a
saving you cannot prove, and you cannot prove one without a protocol both
sides accepted before the traffic moved.

---

## What is defensible

Not the instrument, which is open source. Not the corpus, which anyone can
rebuild for four dollars a cell. Not the index, which can be forked.

**The counterfactual.** Being the party whose declared baseline both sides
accept is what converts a measurement into a billing event, and it is the only
asset here with switching cost: leaving means renegotiating every baseline.

That is why the protocol matters more than the meter, and why it is published
rather than held. A method only one party may use is not a method.
