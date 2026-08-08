"""Should you self-host at all, and against which API.

Everything else in this package assumes the decision is already made. It
answers which accelerator, for a team that already runs their own inference.
That is a small population, and it is small partly because the number needed
to make the decision does not exist: a team can look up what an API charges
per million tokens and cannot compute what the same work would cost them.

This computes the second number and puts both on one axis.

**The axis is cost per compliant token, not cost per token.** A provider whose
median first-token latency exceeds the workload's bound delivers nothing
sellable at any price, so its cost per compliant token is not high, it is
undefined. That distinction is the entire reason a price sheet cannot answer
this question and the reason the comparison is worth making.

**Four things a naive comparison gets wrong**, each handled here:

- **Utilisation.** A rented GPU is paid for whether or not a request arrives.
  An API is paid per token. Below some traffic level self-hosting is more
  expensive at any price per hour, and the break-even is a number rather than
  a judgement.
- **The latency bound.** Compared on compliant output only, both sides.
- **Engineering cost.** Running inference is not free even when the GPU is
  cheap. It is a declared input rather than a hidden assumption.
- **Input tokens.** APIs bill prompt and completion separately and at
  different rates, and prompt tokens dominate most real workloads.

**What this is not.** It is not a recommendation about vendor risk, data
residency, model quality, or how much operational appetite a team has. Those
decide the question at least as often as cost does, and pretending otherwise
would be the kind of answer a price sheet gives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from berth.place import decide


@dataclass(frozen=True)
class ApiOffer:
    """A hosted endpoint's published terms.

    `ttft_p50_ms` and `ttft_p99_ms` are measured or vendor-published latency.
    They are the reason an offer can be excluded: a price with no compliant
    output behind it is not a price.
    """

    provider: str
    model: str
    input_per_mtok: float
    output_per_mtok: float
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    source: str = "published rate card"

    def cost_per_request(self, prompt_tokens: int, output_tokens: int) -> float:
        return (prompt_tokens * self.input_per_mtok / 1e6
                + output_tokens * self.output_per_mtok / 1e6)


@dataclass
class SelfHostOption:
    """What it would cost to run the workload yourself."""

    silicon: str
    devices: int
    nodes: int
    price_per_hour: float
    ttft_ms: float
    tpot_ms: float
    capacity_per_hour: float
    measured: bool
    band: float


@dataclass
class Comparison:
    workload: dict
    requests_per_hour: float
    self_host: SelfHostOption | None
    self_host_cost_per_request: float
    self_host_utilisation: float
    engineering_cost_per_hour: float
    api_offers: list[dict] = field(default_factory=list)
    excluded_offers: list[dict] = field(default_factory=list)
    breakeven_requests_per_hour: float | None = None
    verdict: str = ""
    caveats: list[str] = field(default_factory=list)


def compare(*, model_key: str, prompt_tokens: int, output_tokens: int,
            slo_bound_ms: float, requests_per_hour: float,
            api_offers, concurrency: int = 8,
            engineering_cost_per_hour: float = 0.0,
            fleet=None, prices=None) -> Comparison:
    """Put self-hosting and every API offer on one axis.

    `requests_per_hour` is the load that decides it. A rented GPU is paid for
    idle; an API is not. Two teams with identical workloads and different
    traffic get opposite answers, which is precisely what a price sheet cannot
    express.
    """
    # decide() raises when nothing in the fleet meets the bound, which is the
    # right behaviour there and the wrong one here: the comparison is still
    # worth making against APIs alone, and "no hardware you can rent will do
    # this" is a useful answer rather than an error.
    try:
        rec = decide(workload_class="self-host-vs-api", model_key=model_key,
                     incumbent=_any_key(fleet), slo_bound_ms=slo_bound_ms,
                     batch=concurrency, prompt_tokens=prompt_tokens,
                     output_tokens=output_tokens, fleet=fleet, prices=prices)
        best = rec.candidates[0] if rec.candidates else None
    except SystemExit:
        best = None
    caveats = []

    if best is None:
        sh, sh_cost, util = None, float("inf"), 0.0
        caveats.append(
            "No accelerator in the fleet meets this bound for this workload, "
            "so self-hosting is not an option at any price and the comparison "
            "is between APIs only.")
    else:
        hourly = best["cost_per_mtok"] * 0.0   # placeholder, replaced below
        # Sustainable request rate: one replica at the declared concurrency,
        # each request occupying the node for its own generation time.
        seconds_per_request = (best["ttft_ms"]
                               + best["tpot_ms"] * output_tokens) / 1000.0
        capacity_per_hour = 3600.0 / seconds_per_request * concurrency

        price_hr = _price_of(best, fleet, prices)
        hourly_per_node = price_hr * best["devices"] + engineering_cost_per_hour
        # Capacity is per node, so serving more than one node's worth costs
        # more than one node. An earlier version divided one node's hourly
        # cost by any request rate and reported self-hosting getting
        # arbitrarily cheap with volume, which is the shape of a graph nobody
        # has ever had.
        import math
        nodes = max(1, math.ceil(requests_per_hour / capacity_per_hour)) \
            if capacity_per_hour else 1
        hourly = hourly_per_node * nodes
        util = (requests_per_hour / (capacity_per_hour * nodes)
                if capacity_per_hour else 0.0)
        sh_cost = hourly / requests_per_hour if requests_per_hour else float("inf")

        sh = SelfHostOption(
            silicon=best["silicon"], devices=best["devices"], nodes=nodes,
            price_per_hour=price_hr, ttft_ms=best["ttft_ms"],
            tpot_ms=best["tpot_ms"],
            capacity_per_hour=capacity_per_hour,
            measured=best["measured"], band=best["band"])

        if util > 0.85:
            caveats.append(
                f"At {requests_per_hour:,.0f} requests an hour this placement "
                f"runs at {util:.0%} of capacity, which leaves no headroom for "
                f"a burst. Real traffic is not smooth, and a node with no slack "
                f"misses its bound before it runs out of throughput.")
        if not best["measured"]:
            caveats.append(
                f"The self-hosted figure rests on a spec-sheet prior for "
                f"{best['silicon']} with this model. A prior has been observed "
                f"above 40 percent error, so treat the comparison as a "
                f"hypothesis worth measuring rather than a decision.")

    kept, excluded = [], []
    for o in api_offers:
        cost = o.cost_per_request(prompt_tokens, output_tokens)
        row = {"provider": o.provider, "model": o.model,
               "cost_per_request": cost,
               "cost_per_hour": cost * requests_per_hour,
               "input_per_mtok": o.input_per_mtok,
               "output_per_mtok": o.output_per_mtok,
               "ttft_p99_ms": o.ttft_p99_ms, "source": o.source}
        if o.ttft_p99_ms is not None and o.ttft_p99_ms > slo_bound_ms:
            row["excluded_reason"] = (
                f"p99 first-token latency {o.ttft_p99_ms:.0f} ms exceeds the "
                f"{slo_bound_ms:.0f} ms bound. Cost per compliant token here "
                f"is not high, it is undefined.")
            excluded.append(row)
        elif o.ttft_p99_ms is None:
            row["excluded_reason"] = None
            row["unverified_latency"] = True
            kept.append(row)
            caveats.append(
                f"{o.provider} publishes no latency figure, so it is ranked on "
                f"price alone and cannot be checked against the bound. That is "
                f"the gap this whole comparison exists to close.")
        else:
            kept.append(row)

    kept.sort(key=lambda r: r["cost_per_request"])

    breakeven, floor = None, None
    if sh is not None and kept:
        cheapest_api = kept[0]["cost_per_request"]
        hourly_per_node = (_price_of(best, fleet, prices) * best["devices"]
                           + engineering_cost_per_hour)
        # The cheapest self-hosting can ever be: one node at full capacity.
        # Beyond that you buy another node and the per-request cost stops
        # falling. An earlier version divided the hourly cost by the API price
        # and reported a break-even four times the capacity of the node it was
        # describing, which is the same error as ranking a placement that
        # cannot meet the bound.
        floor = (hourly_per_node / sh.capacity_per_hour
                 if sh.capacity_per_hour else float("inf"))
        if floor >= cheapest_api:
            caveats.insert(0,
                f"Self-hosting cannot beat this API at any volume. At full "
                f"capacity one node delivers {sh.capacity_per_hour:,.0f} "
                f"requests an hour for ${floor*1000:.3f} per thousand, and the "
                f"cheapest compliant API is ${cheapest_api*1000:.3f}. More "
                f"traffic means more nodes, so the per-request cost stops "
                f"falling here.")
        elif cheapest_api > 0:
            breakeven = min(hourly_per_node / cheapest_api,
                            sh.capacity_per_hour)

    if sh is None:
        verdict = "use an API, no placement meets the bound"
    elif not kept:
        verdict = "self-host, no compliant API offer"
    elif sh_cost < kept[0]["cost_per_request"]:
        verdict = (f"self-host on {sh.silicon}, "
                   f"{kept[0]['cost_per_request'] / sh_cost:.1f}x cheaper than "
                   f"the cheapest compliant API at this volume")
    else:
        verdict = (f"use {kept[0]['provider']}, "
                   f"{sh_cost / kept[0]['cost_per_request']:.1f}x cheaper than "
                   f"self-hosting at this volume")

    caveats.append(
        "Cost is one input. Vendor risk, data residency, model quality and "
        "operational appetite decide this at least as often, and none of them "
        "are modelled here.")

    return Comparison(
        workload={"model": model_key, "prompt_tokens": prompt_tokens,
                  "output_tokens": output_tokens, "slo_bound_ms": slo_bound_ms,
                  "concurrency": concurrency},
        requests_per_hour=requests_per_hour, self_host=sh,
        self_host_cost_per_request=sh_cost, self_host_utilisation=util,
        engineering_cost_per_hour=engineering_cost_per_hour,
        api_offers=kept, excluded_offers=excluded,
        breakeven_requests_per_hour=breakeven, verdict=verdict,
        caveats=caveats)


def _any_key(fleet):
    from berth.silicon import FLEET
    return next(iter(fleet or FLEET))


def _price_of(candidate, fleet, prices):
    from berth.silicon import FLEET
    key = candidate["silicon"]
    if prices and key in prices:
        return prices[key]
    return (fleet or FLEET)[key].base_price_hr


def render(c: Comparison) -> str:
    w = c.workload
    out = [
        "SELF-HOST OR API",
        "=" * 68,
        f"  model        {w['model']}",
        f"  workload     {w['prompt_tokens']} prompt, {w['output_tokens']} "
        f"output, concurrency {w['concurrency']}",
        f"  bound        first token under {w['slo_bound_ms']:.0f} ms",
        f"  volume       {c.requests_per_hour:,.0f} requests an hour",
        "",
    ]
    if c.self_host:
        s = c.self_host
        out += [
            f"  SELF-HOST    {s.nodes} node{'s' if s.nodes > 1 else ''} of "
            f"{s.devices}x {s.silicon} at ${s.price_per_hour:.2f}/hr each"
            + (f" plus ${c.engineering_cost_per_hour:.2f}/hr engineering"
               if c.engineering_cost_per_hour else ""),
            f"               ${c.self_host_cost_per_request*1000:.3f} per 1,000 "
            f"requests, running at {c.self_host_utilisation:.0%} of capacity",
            f"               {'MEASURED' if s.measured else f'prior, ±{s.band:.0%}'}",
            "",
        ]
    out += [f"  {'API':<22}{'$/1k req':>12}{'in $/Mtok':>12}"
            f"{'out $/Mtok':>12}{'p99 ttft':>14}"]
    for r in c.api_offers:
        ttft = f"{r['ttft_p99_ms']:.0f}ms" if r.get("ttft_p99_ms") else "unpublished"
        out.append(f"  {r['provider']:<22}{r['cost_per_request']*1000:>12.3f}"
                   f"{r['input_per_mtok']:>12.2f}{r['output_per_mtok']:>12.2f}"
                   f"{ttft:>14}")
    for r in c.excluded_offers:
        out.append(f"  {r['provider']:<22}{'excluded':>12}   {r['excluded_reason']}")

    if c.self_host:
        out += ["", f"  One node saturates at {c.self_host.capacity_per_hour:,.0f} "
                    f"requests an hour. Past that the cost per request stops "
                    f"falling, because the next request buys a second node."]
    if c.breakeven_requests_per_hour:
        out += [f"  Break-even at {c.breakeven_requests_per_hour:,.0f} requests "
                f"an hour. Below that a rented node is paid for while idle and "
                f"the API wins at any price per hour."]
    out += ["", f"  VERDICT      {c.verdict}", ""]
    for cav in c.caveats:
        out.append(f"  note   {cav}")
    return "\n".join(out)
