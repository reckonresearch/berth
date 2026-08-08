"""Watchers. Turn the outside world changing into a trigger.

The agent's loop takes `detect_triggers` as an injected callable. This module
implements it against real sources: model registries, provider price sheets,
and the corpus itself.

Every watcher is a pure function of a fetched payload and a stored state, so
the network layer can be swapped for a fixture and the logic tested without
polling anything. That separation is deliberate: a watcher that can only be
tested against a live API is a watcher that gets tested rarely.

**Polling, not webhooks.** Webhooks need an endpoint we would have to run,
authenticate and keep available, which is infrastructure. Polling a registry
every few hours is enough: a model version that shipped this morning does not
need to move production traffic before lunch, and the cost of being six hours
late is zero.

**What a watcher never does.** Decide anything. It reports that an input
changed. Whether that matters is the estimator's question and whether it is
worth telling anyone is the agent's.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime

from berth.agent import Trigger


class WatchError(RuntimeError):
    """A source could not be read. Never raised into the agent loop: a source
    being down is not a reason to stop watching the others."""


# ------------------------------------------------------------------ fetching

def fetch_json(url: str, timeout: int = 15) -> dict:
    """One HTTP GET returning parsed JSON. Isolated so it can be replaced."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "berth-watch/0.1 (reckonresearch.com)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        raise WatchError(f"{url}: {e}") from e


# ------------------------------------------------------------------- state

@dataclass
class WatchState:
    """What each source looked like last time.

    A watcher with no memory fires on every poll, which is the same failure as
    an agent with no memory: the channel gets muted. The first observation of
    a source is recorded and produces nothing, because we do not know whether
    it changed.
    """

    model_versions: dict[str, str] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    corpus_cells: set[tuple[str, str]] = field(default_factory=set)
    last_polled: dict[str, str] = field(default_factory=dict)
    # A price that moves by less than this is noise. Providers adjust rates by
    # fractions of a percent constantly and none of it changes a placement.
    price_epsilon: float = 0.02

    def note_poll(self, source: str):
        self.last_polled[source] = datetime.now(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------- watchers

def model_revision(payload: dict) -> str | None:
    """The identity of a model's current weights, from a registry payload.

    Hugging Face exposes `sha`, the commit of the model repository. It changes
    when the weights change and does not change when the card is edited, which
    is what makes it the right field: a README fix is not a placement event.
    """
    return payload.get("sha") or payload.get("lastModified")


def watch_models(model_ids, state: WatchState, *, fetch=fetch_json,
                 base="https://huggingface.co/api/models/"):
    """Yield (model_id, old, new) for every model whose weights moved."""
    changed = []
    for mid in model_ids:
        try:
            payload = fetch(f"{base}{mid}")
        except WatchError:
            # A registry being unreachable is not a change. Recording it as one
            # would fire the agent on an outage.
            continue
        rev = model_revision(payload)
        if rev is None:
            continue
        old = state.model_versions.get(mid)
        state.model_versions[mid] = rev
        state.note_poll(f"model:{mid}")
        if old is None:
            continue                      # first sight, nothing to compare
        if old != rev:
            changed.append((mid, old, rev))
    return changed


def watch_prices(price_source, state: WatchState):
    """Yield (silicon, old, new) for every rate that moved beyond the epsilon.

    `price_source` is a callable returning {silicon: dollars_per_hour}, so a
    customer can point this at their own negotiated rates rather than at list
    prices. Their contracted rate is the one that decides their placement, and
    it is usually not the published one.
    """
    changed = []
    try:
        current = price_source()
    except Exception as e:                # noqa: BLE001 - any source, any error
        raise WatchError(f"price source failed: {e}") from e
    for sil, price in current.items():
        old = state.prices.get(sil)
        state.prices[sil] = price
        if old is None or old <= 0:
            continue
        if abs(price - old) / old > state.price_epsilon:
            changed.append((sil, old, price))
    state.note_poll("prices")
    return changed


def watch_corpus(cells, state: WatchState):
    """Yield (silicon, model) for every cell that is newly measured.

    A prior becoming a measurement is the most interesting trigger available,
    because it is the only one where the answer can change without anything in
    the customer's world changing at all. It is also the trigger that makes
    the corpus visible to the person paying for it.
    """
    current = {(s, m) for s, m in cells}
    new = current - state.corpus_cells
    state.corpus_cells = current
    state.note_poll("corpus")
    return sorted(new)


# --------------------------------------------------------- the trigger view

def build_detector(state: WatchState, *, price_source=None, corpus_cells=None,
                   fetch=fetch_json):
    """Return a `detect_triggers(watched) -> [Trigger]` for the agent.

    Polls once per call across all sources, then answers per workload class
    from the cached result. Polling per class would hit a registry once per
    class watching the same model, which is both wasteful and a good way to
    get rate limited.
    """
    fired = {"models": {}, "prices": {}, "corpus": set()}

    def poll(model_ids):
        for mid, _old, new in watch_models(model_ids, state, fetch=fetch):
            fired["models"][mid] = new
        if price_source is not None:
            try:
                for sil, _old, new in watch_prices(price_source, state):
                    fired["prices"][sil] = new
            except WatchError:
                pass
        if corpus_cells is not None:
            fired["corpus"] = set(watch_corpus(corpus_cells, state))

    def detect(watched):
        triggers = []
        if watched.model_id in fired["models"]:
            triggers.append(Trigger.MODEL_VERSION)
        if watched.current_silicon in fired["prices"]:
            triggers.append(Trigger.PRICE_CHANGE)
        if any(m == watched.model_key for _s, m in fired["corpus"]):
            triggers.append(Trigger.CORPUS_CELL)
        return triggers

    detect.poll = poll
    detect.fired = fired
    return detect
