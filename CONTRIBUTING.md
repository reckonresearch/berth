# Contributing

Ground rules that keep the core auditable:

1. **The analytical core stays stdlib-only.** Fitted/learned components layer
   on top; they never replace the closed-form model. Every number in an
   estimate must be derivable by hand.
2. **Physics changes require validation changes.** If you touch
   `berth/estimate.py` or `berth/workload.py`, update `bench/validate.py` in the
   same PR — the term-by-term checks are the spec.
3. **Blind recovery is the test pattern.** New calibration/detection features
   are validated by generating traces from hidden ground truth and proving
   recovery; asserting on outputs you tuned by eye does not count.
4. **Measured data over priors.** Efficiency factors in `FLEET` are priors.
   PRs replacing them must attach the traces (JSONL) and the validate report.
5. Run before pushing: `ruff check . && python -m pytest tests/ -q`.
   Conventional commits (feat:/fix:/chore:/docs:).
