---
name: A measurement
about: A cell you ran, or a number you think is wrong
labels: measurement
---

**Silicon and model**

**Serving stack and exact version** (from the launch command, not from memory)

**The full server command**

**What you observed, against what berth predicted**

**Traces attached?** Anonymised is fine. Mock traces are not.

The serving stack version matters more than it looks. The same card and model
under two vLLM versions has measured over twice the throughput, so a trace
without it is a trace we cannot place.
