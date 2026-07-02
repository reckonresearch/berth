"""PlacementClient: the four primitives.

    profile(workload)          -> ComputeSignature
    estimate(signature)        -> list[Estimate]  (all silicon, tax annotated)
    place(signature, policy)   -> PlacementHandle
    migrate(handle)            -> PlacementHandle (possibly moved)

Design mirror of Tinker: the user's policy is plain Python (an objective
callable over Estimate plus optional constraints), not a config enum. The
platform never decides *what* is optimal — it executes the user's definition
of optimal against live market state. Neutrality lives in that separation.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from .backend import Backend
from .estimate import Estimate
from .estimate import estimate as _estimate
from .workload import ComputeSignature, WorkloadSpec
from .workload import profile as _profile

Objective = Callable[[Estimate], float]        # lower is better
Constraint = Callable[[Estimate], bool]        # False filters the target out


@dataclass(frozen=True)
class PlacementPolicy:
    objective: Objective
    constraints: tuple[Constraint, ...] = ()
    # Migration hysteresis: don't move unless new objective beats current by
    # this fraction. Real migrations cost warm KV caches, image pulls, and
    # connection draining; free-lunch migration models churn themselves to death.
    min_improvement: float = 0.15


# Common objectives, provided for convenience — users can write their own.
def min_cost(e: Estimate) -> float:
    return e.cost_per_mtok


def min_tpot(e: Estimate) -> float:
    return e.tpot_ms


@dataclass
class PlacementHandle:
    allocation_id: str
    silicon: str
    signature: ComputeSignature
    policy: PlacementPolicy
    estimate: Estimate


class PlacementClient:
    def __init__(self, backend: Backend, profile_resolver=None):
        """profile_resolver(silicon, sig) -> SiliconProfile, optional.

        Lets calibration decide which profile applies to *this* workload
        (per-class fits). Pricing still comes from the backend — the resolver
        adjusts efficiency knowledge, not market state.
        """
        self._backend = backend
        self._resolver = profile_resolver

    def profile(self, workload: WorkloadSpec) -> ComputeSignature:
        return _profile(workload)

    def estimate(self, sig: ComputeSignature) -> list[Estimate]:
        """Estimate across the whole fleet at current prices, tax-annotated."""
        results = [
            _estimate(
                sig,
                self._resolver(name, sig) if self._resolver else hw,
                self._backend.price_hr(name),
            )
            for name, hw in self._backend.fleet().items()
        ]
        feasible = [e for e in results if e.feasible]
        if feasible:
            best = min(e.cost_per_mtok for e in feasible)
            results = [
                replace(e, placement_premium=(e.cost_per_mtok / best) - 1.0) if e.feasible else e
                for e in results
            ]
        return sorted(results, key=lambda e: e.cost_per_mtok)

    def place(self, sig: ComputeSignature, policy: PlacementPolicy) -> PlacementHandle:
        choice = self._select(sig, policy)
        if choice is None:
            raise RuntimeError("no feasible placement satisfies the policy constraints")
        alloc = self._backend.bind(choice.silicon, choice.n_devices)
        return PlacementHandle(alloc, choice.silicon, sig, policy, choice)

    def migrate(self, handle: PlacementHandle) -> PlacementHandle:
        """Re-evaluate at current market state; move only past hysteresis."""
        candidate = self._select(handle.signature, handle.policy)
        if candidate is None:
            return handle  # nothing feasible elsewhere; hold position

        # Re-price the CURRENT placement at current market, not at bind-time.
        current = next(
            e for e in self.estimate(handle.signature) if e.silicon == handle.silicon
        )
        violates = current.feasible and any(
            not c(current) for c in handle.policy.constraints)
        if not current.feasible or violates:
            # Infeasible OR in policy-constraint violation: hysteresis does
            # not apply — holding a placement the user's constraints forbid
            # is never the right answer.
            improvement = float("inf")
        else:
            cur_score = handle.policy.objective(current)
            new_score = handle.policy.objective(candidate)
            improvement = (cur_score - new_score) / cur_score if cur_score > 0 else 0.0

        if candidate.silicon == handle.silicon or improvement < handle.policy.min_improvement:
            return replace_handle_estimate(handle, current)

        self._backend.release(handle.allocation_id)
        alloc = self._backend.bind(candidate.silicon, candidate.n_devices)
        return PlacementHandle(alloc, candidate.silicon, handle.signature, handle.policy, candidate)

    def _select(self, sig: ComputeSignature, policy: PlacementPolicy) -> Estimate | None:
        pool = [e for e in self.estimate(sig) if e.feasible]
        for c in policy.constraints:
            pool = [e for e in pool if c(e)]
        return min(pool, key=policy.objective) if pool else None


def replace_handle_estimate(h: PlacementHandle, e: Estimate) -> PlacementHandle:
    return PlacementHandle(h.allocation_id, h.silicon, h.signature, h.policy, e)
