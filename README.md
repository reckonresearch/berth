# berth

**Adaptive placement infrastructure for AI inference.**

Predict what a workload costs and how fast it runs on a given accelerator,
before you rent it.

```bash
pip install berth-placement
berth place --workload-class voice --model llama3-8b --incumbent h100-pcie --slo-ms 800
```

[Docs](https://docs.reckonresearch.com) ·
[The Placement Index](https://reckonresearch.com/index/) ·
[Defect register](https://github.com/reckonresearch/berth/blob/main/DEFECTS.md)

[![ci](https://github.com/reckonresearch/berth/actions/workflows/ci.yml/badge.svg)](https://github.com/reckonresearch/berth/actions)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/reckonresearch/berth/blob/main/LICENSE)

---

## The problem

Compute is sold by the hour. It is consumed as work delivered under a
deadline. No standard unit connects the two.

So there is nothing to be efficient *per*. Tokens per second can double while
a service delivers nothing sellable, because a token that arrives after its
deadline is not a token. Dollars per GPU-hour is the price of the machine, not
the cost of the work.

## The unit

**One output token delivered inside its latency target and above a declared
quality floor.** A served token.

Everything here measures in that unit. The specification is published and
anyone may use it, including people who compete with us. A unit nobody else
may use is not a unit.

## Three tools

**berth** predicts what a workload costs and how fast it runs on a given
accelerator, before you rent it. Closed-form roofline plus a queueing term. No
learned components, roughly a page of arithmetic.

**sounding** measures the real thing and checks the prediction. It never
imports the estimator, enforced by test: evidence that cannot be separated
from the thing it evaluates is not evidence.

**pilot** watches for change, moves the workload, and proves what the move
saved. It routes at the workload level, by programming your infrastructure
rather than by carrying your traffic.

## What we found

Fit the decode constant on one accelerator, then predict a different
accelerator it has never seen, on different memory technology.

**Worst held-out fold: 9.5 percent, against a 15 percent gate published before
the first run.**

The residual is the real spread between the two cards. One constant across
both costs about nine percent, which is the price of generality and is stated
rather than fitted away.

Two more from the same corpus:

**Prefill admission is serial, and not as a quirk of one server.** Effective
parallelism 1.09 under vLLM and 1.01 under SGLang. Modelling that one term
drops first-token error from roughly 65 percent to under 9.

**Decode is stack-independent to three decimals.** 0.854 under SGLang against
0.850 under vLLM, same card, same model.

Every trace is downloadable. Every number is reproducible.

## Quick start

```python
from berth import MODELS, WorkloadSpec, profile
from berth.place import decide, render

print(render(decide(workload_class="voice", model_key="llama3-8b",
                    incumbent="h100-pcie", slo_bound_ms=800, batch=8,
                    prompt_tokens=512, output_tokens=128)))
```

A placement that misses your bound is excluded rather than ranked cheaply. Its
cost per compliant token is not high, it is undefined.

## Command line

| | |
| --- | --- |
| `berth estimate` | one workload on one accelerator, or all of them |
| `berth place` | a decision record: what is recommended, what it beat, by how much |
| `berth premium` | rank placements and show the premium for choosing wrong |
| `berth versus` | self-host or API, on one axis |
| `berth pilot` | one pass of the agent. Shadow by default |
| `berth holdout` | check a holdout assignment before opening a period |

## Adaptive placement

Deciding which chip, which provider, and at which price a model runs under a
stated p99, then moving it as prices, capacity, and traffic shift.

Three things make it adaptive.

**It is workload-conditional.** The answer depends on your token lengths and
your concurrency, not on a benchmark average. The right chip can invert inside
a single workload as concurrency changes.

**It is bounded by your service level.** A placement that misses your p99 has
no price, not a low one.

**It constantly adapts.** Prices move, models ship, capacity changes. So does
the answer, and most teams decide once and never look again.

`pilot` closes that loop. It watches model registries, serving-stack releases,
provider prices and the corpus; re-estimates when something moves; and, where
your declared autonomy policy permits, moves the workload.

The change lands as a pull request against your deployment config, carrying
the diff and the evidence. Under a policy you committed in advance, pilot
merges it and your delivery pipeline moves the workload. Without one, it opens
the pull request and stops.

It cannot write to a default branch, cannot touch a path you did not declare,
and cannot merge without the policy that permitted it recorded in the merge
commit. Reverting a move is reverting a commit, and deleting
`.berth/classes.yaml` stops everything.

Most passes produce nothing, and that is the design. An agent that proposes
every week gets muted, and a muted agent is worse than none because it looks
like coverage.

See [the docs](https://docs.reckonresearch.com/pilot/).

## Proving a saving

A better estimate is not a saving. Proving one needs a declared baseline, a
held-out slice of traffic, and an agreed method for comparing them.

That is the [Placement Holdout
Protocol](https://docs.reckonresearch.com/holdout/), published under CC BY so
a counterparty can check it before agreeing to be measured.

## Contributing a measurement

The corpus grows by measurement. If you run a cell we have not,
[send it](https://docs.reckonresearch.com/verify-and-contribute/).

Six cells across three memory technologies and two vendors are measured today.
Everything else in the fleet is a spec-sheet prior, and every line the CLI
prints says which.

## When it is wrong

[`DEFECTS.md`](https://github.com/reckonresearch/berth/blob/main/DEFECTS.md) is the register: fourteen instrument failures, what
caused each, how it was caught, and the test that fails if it returns.

Four let bad data through. **Seven rejected good data**, which is the more
dangerous direction, because a false negative is caught downstream eventually
and a false positive is silent. One of them nearly discarded the most valuable
cell in the corpus. **Three were correct code that nothing could reach**, which
no unit test can catch.

Only physical impossibility refuses a file. Everything else prints and passes.

A measurement tool that has never been caught lying has not been used hard
enough.

---

Apache-2.0. Built by [Reckon Research](https://reckonresearch.com).
