# Contributing

The most valuable contribution is a measurement.

## A cell nobody has run

A cell is one accelerator, one model, one serving stack, one precision. Six
exist today. If you have access to hardware outside that set, a sweep takes
about an hour and a few dollars of compute.

```
pip install berth-placement
./bench/p0_run.sh <silicon> <model> <model-id>
```

**Where to send it.** Open a pull request adding the file to
`data/contributed/`. CI runs `python -m bench.check_contributed` and rejects
any file containing a mock record or any record without explicit provenance.
If a pull request is awkward, open an issue with the
[measurement template](https://github.com/reckonresearch/berth/issues/new?template=measurement.md)
and attach it, or email traces@reckonresearch.com.

Send the run conditions alongside the traces. `sounding` writes them to
`traces.jsonl.meta.json` and prints a reminder, because a trace without its
conditions is a number without a denominator.

**The exact server command matters more than it looks.** The same card and
model under two vLLM versions has measured over twice the throughput, so a
trace without the launch command is a trace we cannot place.

**What happens then.** The gate runs in CI, so you find out in minutes rather
than waiting on us. A cell that passes is published with your attribution and
its provenance, and it enters the corpus that every estimate is checked
against. A cell that fails tells you which line and why.

The harness records what it can verify and refuses to guess at the rest. It
reads hardware identity where the server is local, verifies the served model
against the endpoint, cross-checks precision against the launch flags, and
probes whether prefix caching is on. Where it cannot verify something it
records that it could not, rather than filling in a plausible value.

## A number you think is wrong

Open an issue with the model, the concurrency, the prompt length distribution,
and what you observed against what berth predicted. One data point with
context is worth more than a general objection, and it is the fastest route
into `DEFECTS.md`.

Several of the registered defects came from someone running the tool and
reporting what happened, and the most useful of those reported a pattern
across three attempts rather than a single failure.

## Code

```
ruff check . && python -m pytest tests/ -q
```

Five rules specific to this project.

**The analytical core stays stdlib-only.** Fitted and learned components layer
above it, never inside it.

**Physics changes require validation changes.** A term that changes a
prediction changes what the validation record claims.

**Blind recovery is the test pattern.** New calibration features are proven by
generating traces from hidden ground truth and recovering it.

**Every number carries its provenance.** MEASURED, FITTED, CONFIG, SIM or
HYPOTHESIS. A number labelled measured that contains a simulated input is the
one unrecoverable error here.

**A guard needs a test for what it permits, not only for what it blocks.**
Most of the registered defects were checks that rejected correct data, and
each had a test for the failure path and none for the success path.
