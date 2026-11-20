from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from CVRP.cvrp_instance_and_load import CVRPInstance, euclidean_2d_edge_distance
from CVRP.cvrp_solution import Solution, is_valid
from CVRP.initial_solution import initial_solution_cws


@dataclass(frozen=True)
class SearchResult:
    solution: Solution
    initial_cost: float
    cost: float
    iterations: int
    elapsed_seconds: float


_EPS = 1e-12
_CHECKPOINT_SECONDS = (0, 1, 2, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360 , 420 , 480, 540, 599)
#_CHECKPOINT_LOG =   Path (r"D:\Users\Igor\Desktop\Praksa\__Projekat\logs\gvns_log.txt")

#google collab checkpoint
_CHECKPOINT_LOG = ( Path(__file__).resolve().parent / "data" / "logs" / "gvns_log.txt")


instance: CVRPInstance
CandidateGenerator = Callable[[Solution], Iterable[Solution]]
deadline: float | None = None


def _is_deadline() -> bool:
    """Return True when the search deadline has expired."""
    return deadline is not None and time.monotonic() >= deadline


def _route_nodes(route: Sequence[int]) -> list[int]:
    """Translate depot sentinel 0 using the passed instance."""
    cvrp = instance
    if cvrp is None:
        raise RuntimeError("Pass a CVRP instance or set the module-level instance first")
    depot = cvrp.depot_ids[0]
    return [depot if node == 0 else node for node in route]


def route_distance(route: Sequence[int]) -> float:
    """Return Euclidean distance using the passed instance."""
    cvrp = instance
    if cvrp is None:
        raise RuntimeError("Pass a CVRP instance or set the module-level instance first")
    nodes = _route_nodes(route)
    return sum(euclidean_2d_edge_distance(cvrp.coordinates, a, b) for a, b in zip(nodes, nodes[1:]))


def objective(solution: Solution | Sequence[Sequence[int]]) -> float:
    """Total route distance using the passed  instance."""
    routes = solution.routes if isinstance(solution, Solution) else solution
    return sum(route_distance(route) for route in routes)


def _route_demand(route: Sequence[int]) -> int:
    return sum(instance.demands[customer] for customer in route[1:-1])


def _copy_solution(solution: Solution) -> Solution:
    return Solution([route.copy() for route in solution.routes])


def _relocation_candidates(solution: Solution) -> Iterable[Solution]:
    """Yield all feasible one-customer relocate moves (including new routes)."""
    for source_index, source in enumerate(solution.routes):
        if _is_deadline():
            return
        for source_position in range(1, len(source) - 1):
            if _is_deadline():
                return
            customer = source[source_position]
            base_routes = [route.copy() for route in solution.routes]
            base_routes[source_index].pop(source_position)
            if len(base_routes[source_index]) == 2:
                base_routes.pop(source_index)
            base = Solution(base_routes)
            for target_index, target in enumerate(base.routes):
                if _is_deadline():
                    return
                if _route_demand(target) + instance.demands[customer] > instance.capacity:
                    continue
                for target_position in range(1, len(target)):
                    if _is_deadline():
                        return
                    routes = [route.copy() for route in base.routes]
                    routes[target_index].insert(target_position, customer)
                    candidate = Solution(routes)
                    if is_valid(candidate, instance):
                        yield candidate
            candidate = Solution([*base.routes, [0, customer, 0]])
            if is_valid(candidate, instance):
                yield candidate


def _exchange_candidates(solution: Solution) -> Iterable[Solution]:
    """Yield capacity-feasible swaps of two distinct customers."""
    locations = [
        (route_index, position)
        for route_index, route in enumerate(solution.routes)
        for position in range(1, len(route) - 1)
    ]
    for index, (route_a, position_a) in enumerate(locations):
        if _is_deadline():
            return
        for route_b, position_b in locations[index + 1:]:
            if _is_deadline():
                return
            routes = [route.copy() for route in solution.routes]
            routes[route_a][position_a], routes[route_b][position_b] = (
                routes[route_b][position_b],
                routes[route_a][position_a]
            )
            candidate = Solution([route for route in routes if len(route) > 2])
            if is_valid(candidate, instance):
                yield candidate


def _best_neighbor(solution: Solution,candidates: CandidateGenerator) -> Solution:
    best = solution
    best_cost = objective(solution)
    for candidate in candidates(solution):
        if _is_deadline():
            break
        candidate_cost = objective(candidate)
        if candidate_cost + _EPS < best_cost:
            best, best_cost = candidate, candidate_cost
    return best


def insertion_local_search(solution: Solution) -> Solution:
    """Repeat best-improvement relocate moves until none improves the solution."""
    current = solution

    while not _is_deadline():
        improved = _best_neighbor(current, _relocation_candidates)
        if objective(improved) + _EPS >= objective(current):
            return current
        current = improved

    return current


def exchange_local_search(solution: Solution) -> Solution:
    """Repeat best-improvement exchange moves until none improves the solution."""
    current = solution

    while not _is_deadline():
        improved = _best_neighbor(current, _exchange_candidates)
        if objective(improved) + _EPS >= objective(current):
            return current
        current = improved

    return current

def _write_checkpoint(instance_name: str, seed: int | None, time_limit: float, checkpoint_second: int,
    elapsed_seconds: float, best_cost: float, current_cost: float, iterations: int) -> None:
    _CHECKPOINT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _CHECKPOINT_LOG.open("a", encoding="utf-8") as log_file:
        if checkpoint_second == 0:
            print(
                f"\n=== GVNS run: instance={instance_name}, seed={seed}, "
                f"time_limit={time_limit:g}s ===",
                file=log_file,
            )

        print(
            f"[checkpoint {checkpoint_second}s; "
            f"actual elapsed={elapsed_seconds:.2f}s]",
            file=log_file,
        )
        print(f"best_objective: {best_cost:.6f}", file=log_file)
        print(f"current_objective: {current_cost:.6f}", file=log_file)
        print(f"iterations: {iterations}", file=log_file)
        print(file=log_file)


def _shake(solution: Solution, neighborhood: int, rng: random.Random) -> Solution | None:
    """Choose a random feasible relocate (k=1) or exchange (k=2) neighbor."""
    if neighborhood == 1:
        candidates = _relocation_candidates(solution)
    elif neighborhood == 2:
        candidates = _exchange_candidates(solution)
    else:
        raise ValueError("neighborhood must be 1 or 2")

    selected: Solution | None = None
    count = 0

    for candidate in candidates:
        if _is_deadline():
            return None
        if candidate.routes == solution.routes:
            continue

        count += 1
        if rng.randrange(count) == 0:
            selected = candidate

    return selected


def gvns(cvrp: CVRPInstance, time_limit: float = 480.0, seed: int = 0) -> SearchResult:
    """Run GVNS with Clarke-Wright initialization inside the time limit."""
    global instance, deadline
    instance = cvrp

    start = time.monotonic()
    deadline = start + max(0.0, float(time_limit))

    if 0 in instance.customer_ids:
        raise ValueError("customer ID 0 conflicts with the solution depot sentinel")

    initial = initial_solution_cws(instance)
    initial_cost = objective(initial)
    if not is_valid(initial, instance):
        raise ValueError("Clarke-Wright produced an infeasible initial solution")

    rng = random.Random(seed)
    current = _copy_solution(initial)
    current_cost = initial_cost
    best = _copy_solution(current)
    best_cost = current_cost
    iterations = 0
    checkpoint_index = 0

    _write_checkpoint(
        cvrp.name,
        seed,
        time_limit,
        _CHECKPOINT_SECONDS[checkpoint_index],
        time.monotonic() - start,
        best_cost,
        current_cost,
        iterations,
    )
    checkpoint_index += 1

    k_max = 2
    while not _is_deadline():
        k = 1
        while k <= k_max and not _is_deadline():
            shaken = _shake(current, k, rng)
            if shaken is None:
                k += 1
                iterations += 1
                continue

            candidate = _vnd(shaken)
            candidate_cost = objective(candidate)
            if candidate_cost + _EPS < current_cost:
                current, current_cost = candidate, candidate_cost
                k = 1
            else:
                k += 1

            if current_cost + _EPS < best_cost:
                best, best_cost = _copy_solution(current), current_cost
            iterations += 1

        elapsed_seconds = time.monotonic() - start
        while (
            checkpoint_index < len(_CHECKPOINT_SECONDS)
            and elapsed_seconds >= _CHECKPOINT_SECONDS[checkpoint_index]
        ):
            checkpoint_second = _CHECKPOINT_SECONDS[checkpoint_index]
            _write_checkpoint(
                cvrp.name,
                seed,
                time_limit,
                checkpoint_second,
                elapsed_seconds,
                best_cost,
                current_cost,
                iterations,
            )
            checkpoint_index += 1

    return SearchResult(
        solution=best,
        initial_cost=initial_cost,
        cost=best_cost,
        iterations=iterations,
        elapsed_seconds=time.monotonic() - start,
    )

def _vnd(solution: Solution) -> Solution:
    """Run sequential best-improvement VND over relocate and exchange."""
    current = solution
    neighborhood = 1
    neighborhood_max = 2

    while neighborhood <= neighborhood_max and not _is_deadline():
        candidate = (
            insertion_local_search(current)
            if neighborhood == 1
            else exchange_local_search(current)
        )

        if objective(candidate) + _EPS < objective(current):
            current = candidate
            neighborhood = 1
        else:
            neighborhood += 1

    return current


