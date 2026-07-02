"""Execution backends.

The client speaks a minimal backend protocol: current prices, capacity, and
bind/release of placements. SimBackend implements it with a seeded random-walk
spot market — deterministic, testable, and sufficient to exercise the migrate
primitive. A production backend (real orchestrator: Kubernetes + device
plugins, or a broker across neoclouds) implements the same protocol; the
client and policy code do not change. That contract is the point.
"""

import random
from dataclasses import dataclass

from .silicon import FLEET, SiliconProfile


class Backend:
    """Protocol. Subclass and implement these four methods."""

    def fleet(self) -> dict[str, SiliconProfile]:
        raise NotImplementedError

    def price_hr(self, silicon: str) -> float:
        raise NotImplementedError

    def bind(self, silicon: str, n_devices: int) -> str:
        """Reserve devices; return an allocation id. Fail fast if impossible."""
        raise NotImplementedError

    def release(self, allocation_id: str) -> None:
        raise NotImplementedError


@dataclass
class _Allocation:
    silicon: str
    n_devices: int


class SimBackend(Backend):
    """Seeded spot market: prices random-walk each tick(), capacity is finite."""

    def __init__(self, seed: int = 0, capacity_per_class: int = 64, drift: float = 0.06,
                 fleet: dict[str, SiliconProfile] | None = None):
        self._fleet = fleet if fleet is not None else FLEET
        self._rng = random.Random(seed)
        self._mult: dict[str, float] = {name: 1.0 for name in self._fleet}
        self._capacity: dict[str, int] = {name: capacity_per_class for name in self._fleet}
        self._allocs: dict[str, _Allocation] = {}
        self._drift = drift
        self._next_id = 0

    def fleet(self) -> dict[str, SiliconProfile]:
        return self._fleet

    def price_hr(self, silicon: str) -> float:
        return self._fleet[silicon].base_price_hr * self._mult[silicon]

    def bind(self, silicon: str, n_devices: int) -> str:
        if self._capacity[silicon] < n_devices:
            raise RuntimeError(
                f"insufficient capacity: {silicon} has {self._capacity[silicon]}, need {n_devices}"
            )
        self._capacity[silicon] -= n_devices
        self._next_id += 1
        aid = f"alloc-{self._next_id}"
        self._allocs[aid] = _Allocation(silicon, n_devices)
        return aid

    def release(self, allocation_id: str) -> None:
        a = self._allocs.pop(allocation_id)  # KeyError = loud failure, intended
        self._capacity[a.silicon] += a.n_devices

    # --- simulation controls ---
    def tick(self, n: int = 1) -> None:
        """Advance the market: multiplicative random walk, clamped to [0.4, 3.0]."""
        for _ in range(n):
            for name in self._mult:
                shock = self._rng.gauss(0.0, self._drift)
                self._mult[name] = min(3.0, max(0.4, self._mult[name] * (1.0 + shock)))

    def shock(self, silicon: str, multiplier: float) -> None:
        """Force a price event (spot spike, capacity crunch) for demos/tests."""
        self._mult[silicon] = multiplier
