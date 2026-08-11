"""Tests for the self-host versus API comparison.

The comparison changes who the product is for, so the properties that matter
are the ones a price sheet gets wrong: a rented node is paid for idle, a node
has finite capacity, and an offer that misses the bound has no price at all.
"""

import pytest

from berth.versus import ApiOffer, compare, render

GROQ = ApiOffer("Groq", "llama-3-8b", 0.05, 0.08, ttft_p99_ms=260)
SLOW = ApiOffer("Slow", "llama-3-8b", 0.01, 0.01, ttft_p99_ms=4000)
QUIET = ApiOffer("Quiet", "llama-3-8b", 0.02, 0.03)


def _c(rph, offers=(GROQ,), eng=0.0, **kw):
    base = dict(model_key="llama3-8b", prompt_tokens=1469, output_tokens=128,
                slo_bound_ms=800, concurrency=8)
    base.update(kw)
    return compare(requests_per_hour=rph, api_offers=list(offers),
                   engineering_cost_per_hour=eng, **base)


def test_an_offer_missing_the_bound_is_excluded_not_ranked_cheaply():
    """The cheapest offer here is the one that cannot serve the workload.
    Cost per compliant token with no compliant output is undefined, and
    ranking it first is exactly what a price sheet does."""
    c = _c(10_000, offers=(GROQ, SLOW))
    assert [o["provider"] for o in c.api_offers] == ["Groq"]
    assert c.excluded_offers[0]["provider"] == "Slow"
    assert "undefined" in c.excluded_offers[0]["excluded_reason"]


def test_an_unpublished_latency_is_ranked_but_flagged():
    """That gap is the reason this comparison exists, so it is named rather
    than silently trusted or silently dropped."""
    c = _c(10_000, offers=(GROQ, QUIET))
    assert any(o["provider"] == "Quiet" for o in c.api_offers)
    assert any("publishes no latency" in x for x in c.caveats)


def test_self_hosting_gets_cheaper_with_volume_then_stops():
    """A rented node is paid for idle, so cost per request falls with load.
    It stops falling at capacity, because the next request buys a node."""
    costs = [_c(r).self_host_cost_per_request for r in (1_000, 10_000, 40_000)]
    assert costs[0] > costs[1] > costs[2]
    # Past saturation it plateaus rather than continuing down.
    far = _c(2_000_000).self_host_cost_per_request
    assert far == pytest.approx(costs[2], rel=0.05)


def test_capacity_forces_more_nodes_rather_than_a_cheaper_node():
    """The bug this replaced reported self-hosting getting arbitrarily cheap
    with volume, which is the shape of a graph nobody has ever had."""
    small, large = _c(20_000), _c(250_000)
    assert small.self_host.nodes == 1
    assert large.self_host.nodes > 1
    assert large.self_host_cost_per_request >= small.self_host_cost_per_request * 0.5


def test_engineering_cost_can_decide_the_answer_on_its_own():
    """Running inference is not free even when the GPU is cheap. It is a
    declared input rather than a hidden assumption, and at typical volumes it
    is the larger term."""
    free = _c(40_000, eng=0.0)
    staffed = _c(40_000, eng=12.0)
    assert "self-host" == free.verdict[:9]
    assert staffed.verdict.startswith("use Groq")


def test_the_verdict_flips_with_volume_and_nothing_else():
    """Two teams with identical workloads and different traffic get opposite
    answers. That is precisely what a price sheet cannot express."""
    # 40,000 saturates one node, which is where self-hosting is cheapest.
    low, high = _c(1_000), _c(40_000)
    assert low.verdict.startswith("use Groq")
    assert high.verdict.startswith("self-host")


def test_running_at_capacity_is_flagged_as_having_no_headroom():
    """Real traffic is not smooth, and a node with no slack misses its bound
    before it runs out of throughput."""
    c = _c(40_000)
    assert c.self_host_utilisation > 0.85
    assert any("headroom" in x for x in c.caveats)


def test_no_compliant_placement_means_the_comparison_is_apis_only():
    c = _c(10_000, slo_bound_ms=5)
    assert c.self_host is None
    assert "no placement meets the bound" in c.verdict


def test_cost_is_never_presented_as_the_whole_answer():
    """Vendor risk, data residency and model quality decide this at least as
    often, and a tool that implies otherwise is giving a price-sheet answer."""
    c = _c(10_000)
    assert any("Cost is one input" in x for x in c.caveats)


def test_prompt_tokens_are_billed_and_they_dominate():
    """APIs bill prompt and completion separately, and prompt tokens dominate
    most real workloads. Ignoring the input side understates an API by more
    than any placement decision is worth."""
    short = _c(10_000, prompt_tokens=100)
    long = _c(10_000, prompt_tokens=8000)
    assert (long.api_offers[0]["cost_per_request"]
            > short.api_offers[0]["cost_per_request"] * 5)


def test_render_names_the_saturation_point():
    c = _c(10_000)
    text = render(c)
    assert "saturates at" in text
    assert "VERDICT" in text
