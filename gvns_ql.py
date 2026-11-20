"""Q-learning guided General Variable Neighborhood Search for CVRP.

The search structure and Q-learning parameter lists follow Öztop et al.,
"A Novel General Variable Neighborhood Search through Q-Learning for No-Idle
Flowshop Scheduling" (CEC 2020). Flowshop permutations are adapted here to
capacitated vehicle routes; see the documented CVRP-specific adaptations below.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from CVRP.cvrp_instance_and_load import CVRPInstance, euclidean_2d_edge_distance
from CVRP.cvrp_solution import Solution, is_valid
from CVRP.initial_solution import initial_solution_cws


@dataclass(frozen=True)
class SearchParameters:
    epsilon: float
    tau_p: float
    d_max: int
    learning_factor: float
    discount_factor: float


@dataclass(frozen=True)
class SearchResult:
    solution: Solution
    initial_cost: float
    cost: float
    iterations: int
    elapsed_seconds: float
    q_values: dict[str, tuple[float, ...]]


_ACTIONS: dict[str, tuple[float | int, ...]] = {
    "epsilon": (0.05, 0.10, 0.15, 0.20),
    "tau_p": (0.1, 0.2, 0.3, 0.4, 0.5),
    "d_max": (2, 3, 4, 5),
    "learning_factor": (0.2, 0.4, 0.6, 0.8, 1.0),
    "discount_factor": (0.2, 0.4, 0.6, 0.8, 1.0),
}
_EPS = 1e-12
_CHECKPOINT_SECONDS = (0, 1, 2, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360 , 420 , 480, 540, 599)
#_CHECKPOINT_LOG = Path (r"D:\Users\Igor\Desktop\Praksa\__Projekat\logs\gvns_ql_log.txt")

#google collab checkpoint
_CHECKPOINT_LOG = ( Path(__file__).resolve().parent / "data" / "logs" / "gvns_ql_log.txt")

instance: CVRPInstance
CandidateGenerator = Callable[..., Iterable[Solution]]
deadline: float | None = None


def _is_deadline() -> bool:
    """Return True when the search deadline has expired."""
    return deadline is not None and time.monotonic() >= deadline


def _route_nodes(route: Sequence[int], cvrp_instance: CVRPInstance | None = None) -> list[int]:
    """Translate depot sentinel 0 using the passed or module-level instance."""
    cvrp = cvrp_instance if cvrp_instance is not None else instance
    if cvrp is None:
        raise RuntimeError("Pass a CVRP instance or set the module-level instance first")
    depot = cvrp.depot_ids[0]
    return [depot if node == 0 else node for node in route]


def route_distance(route: Sequence[int], cvrp_instance: CVRPInstance | None = None) -> float:
    """Return Euclidean distance using the passed or module-level instance."""
    cvrp = cvrp_instance if cvrp_instance is not None else instance
    if cvrp is None:
        raise RuntimeError("Pass a CVRP instance or set the module-level instance first")
    nodes = _route_nodes(route, cvrp)
    return sum(
        euclidean_2d_edge_distance(cvrp.coordinates, a, b)
        for a, b in zip(nodes, nodes[1:])
    )


def objective(solution: Solution | Sequence[Sequence[int]], cvrp_instance: CVRPInstance | None = None) -> float:
    """Total route distance using the passed or module-level instance."""
    routes = solution.routes if isinstance(solution, Solution) else solution
    return sum(route_distance(route, cvrp_instance) for route in routes)


def _customers(solution: Solution) -> list[int]:
    return [customer for route in solution.routes for customer in route[1:-1]]


def _route_demand(route: Sequence[int]) -> int:
    return sum(instance.demands[customer] for customer in route[1:-1])


def _copy_solution(solution: Solution) -> Solution:
    return Solution([route.copy() for route in solution.routes])


def _without_customers(solution: Solution, removed: set[int]) -> Solution:
    routes = [[0, *(customer for customer in route[1:-1] if customer not in removed), 0] for route in solution.routes]
    return Solution([route for route in routes if len(route) > 2])


def _insert_customer_best(partial: Solution, customer: int) -> Solution:
    """Place one customer at its cheapest feasible position or a new route."""
    best = Solution([*partial.routes, [0, customer, 0]])
    best_cost = objective(best)
    for route_index, route in enumerate(partial.routes):
        if _is_deadline():
            return best
        if _route_demand(route) + instance.demands[customer] > instance.capacity:
            continue
        for position in range(1, len(route)):
            if _is_deadline():
                return best
            routes = [existing.copy() for existing in partial.routes]
            routes[route_index].insert(position, customer)
            candidate = Solution(routes)
            candidate_cost = objective(candidate)
            if candidate_cost + _EPS < best_cost:
                best, best_cost = candidate, candidate_cost
    return best


def _relocation_candidates(solution: Solution, removed: set[int] | None = None) -> Iterable[Solution]:
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
                    if is_valid(candidate, instance, removed):
                        yield candidate
            candidate = Solution([*base.routes, [0, customer, 0]])
            if is_valid(candidate, instance, removed):
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


def _best_neighbor(solution: Solution, candidates: CandidateGenerator, removed: set[int] | None = None) -> Solution:
    best = solution
    best_cost = objective(solution)
    for candidate in candidates(solution, removed):
        if _is_deadline():
            break
        candidate_cost = objective(candidate)
        if candidate_cost + _EPS < best_cost:
            best, best_cost = candidate, candidate_cost
    return best


def insertion_local_search(solution: Solution, removed: set[int] | None = None) -> Solution:
    """Repeated best-improvement relocate search, with restart after gains."""
    current = solution

    while not _is_deadline():
        improved = _best_neighbor(current, _relocation_candidates, removed)
        if objective(improved) + _EPS >= objective(current):
            return current
        current = improved
    return current


def _reference_insertion(solution: Solution, reference: Solution) -> Solution:
    """RIS: insert reference-order customers into their best feasible positions."""
    current = solution
    for customer in _customers(reference):
        if _is_deadline():
            return current
        if customer not in _customers(current):
            continue
        partial = _without_customers(current, {customer})
        candidate = _insert_customer_best(partial, customer)
        if objective(candidate) + _EPS < objective(current):
            current = candidate
    return current


def _local_search_vnd(solution: Solution, reference: Solution) -> Solution:
    """Paper's Local Search_VND: RIS then insertion local search, sequentially."""
    current = solution
    neighborhood = 1
    u_max = 2

    while neighborhood <= u_max and not _is_deadline():
        candidate = (
            _reference_insertion(current, reference)
            if neighborhood == 1
            else insertion_local_search(current)
        )
        if objective(candidate) + _EPS < objective(current):
            current, neighborhood = candidate, 1
        else:
            neighborhood += 1
    return current


def _variable_ig(solution: Solution, d_max: int, rng: random.Random, reference: Solution) -> Solution:
    """VIGALL adaptation: destroy, improve partial routes, reinsert, refine."""
    current = solution
    d = 1
    while d <= min(d_max, len(_customers(current))) and not _is_deadline():
        removed = set(rng.sample(_customers(current), d))
        partial = _without_customers(current, removed)
        #partial = insertion_local_search(partial,removed)
        if _is_deadline():
            return current
        for customer in rng.sample(list(removed), len(removed)):
            if _is_deadline():
                return current
            partial = _insert_customer_best(partial, customer)
        candidate = _local_search_vnd(partial, reference)
        if objective(candidate) + _EPS < objective(current):
            current, d = candidate, 1
        else:
            d += 1
    return current


def _best_block_relocation(solution: Solution, block_size: int) -> Solution:
    """Best feasible relocation of a consecutive customer block."""
    best = solution
    best_cost = objective(solution)
    for source_index, source in enumerate(solution.routes):
        if _is_deadline():
            return best
        for start in range(1, len(source) - block_size):
            if _is_deadline():
                return best
            block = source[start: start + block_size]
            block_demand = sum(instance.demands[customer] for customer in block)
            routes_without = [route.copy() for route in solution.routes]
            del routes_without[source_index][start: start + block_size]
            if len(routes_without[source_index]) == 2:
                routes_without.pop(source_index)
            base = Solution(routes_without)
            #base = insertion_local_search(base, set(block))
            for target_index, target in enumerate(base.routes):
                if _is_deadline():
                    return best
                if _route_demand(target) + block_demand > instance.capacity:
                    continue
                for position in range(1, len(target)):
                    if _is_deadline():
                        return best
                    routes = [route.copy() for route in base.routes]
                    routes[target_index][position:position] = block
                    candidate = Solution(routes)
                    candidate_cost = objective(candidate)
                    if candidate_cost + _EPS < best_cost and is_valid(candidate, instance):
                        best, best_cost = candidate, candidate_cost
            if _is_deadline():
                return best
            candidate = Solution([*base.routes, [0, *block, 0]])
            candidate_cost = objective(candidate)
            if candidate_cost + _EPS < best_cost and is_valid(candidate, instance):
                best, best_cost = candidate, candidate_cost
    return best


def _variable_block_insertion(solution: Solution, d_max: int, reference: Solution) -> Solution:
    """VBIH adaptation with best block relocation and reset-on-improvement."""
    current = solution
    block_size = 1
    while block_size <= min(d_max, max((len(route) - 2 for route in current.routes), default=0)) and not _is_deadline():

        candidate = _best_block_relocation(current, block_size)
        candidate = _local_search_vnd(candidate, reference)
        if objective(candidate) + _EPS < objective(current):
            current, block_size = candidate, 1
        else:
            block_size += 1
    return current


def _shake(solution: Solution, neighborhood: int, rng: random.Random) -> Solution:
    """Randomly select one feasible relocation (k=1) or exchange (k=2)."""
    candidates = (
        _relocation_candidates(solution)
        if neighborhood == 1
        else _exchange_candidates(solution)
    )
    # Reservoir sampling avoids materializing a potentially large neighborhood.
    selected: Solution | None = None
    count = 0
    for candidate in candidates:
        if _is_deadline():
            return solution
        count += 1
        if rng.randrange(count) == 0:
            selected = candidate

    if selected is None:
        print('_shake returned an empty solution.', file=sys.stderr, flush=True)
    return selected if selected is not None else solution


def _random_action_indices(rng: random.Random) -> dict[str, int]:
    return {name: rng.randrange(len(actions)) for name, actions in _ACTIONS.items()}


def _greedy_action_indices(q_table: dict[str, list[float]], rng: random.Random) -> dict[str, int]:
    """Greedy per-parameter action choice; break equal-Q ties at random."""
    selected: dict[str, int] = {}
    for name, values in q_table.items():
        maximum = max(values)
        ties = [index for index, value in enumerate(values) if value == maximum]
        selected[name] = rng.choice(ties)
    return selected


def _parameters_from_indices(indices: dict[str, int]) -> SearchParameters:
    return SearchParameters(
        epsilon=float(_ACTIONS["epsilon"][indices["epsilon"]]),
        tau_p=float(_ACTIONS["tau_p"][indices["tau_p"]]),
        d_max=int(_ACTIONS["d_max"][indices["d_max"]]),
        learning_factor=float(_ACTIONS["learning_factor"][indices["learning_factor"]]),
        discount_factor=float(_ACTIONS["discount_factor"][indices["discount_factor"]]),
    )


def _q_learning_update(q_table: dict[str, list[float]], selected: dict[str, int],
                       reward: float, learning_factor: float, discount_factor: float) -> None:
    """Apply the paper's one-state Q-learning update for each chosen action."""
    for name, action_index in selected.items():
        values = q_table[name]
        old_value = values[action_index]
        values[action_index] = old_value + learning_factor * (reward + discount_factor * max(values) - old_value)


def _log_best(instance_name: str, seed: int, elapsed_seconds: float,
              current_cost: float , best_cost: float, iterations: int) -> None:
    print(
        f"instance={instance_name},seed={seed},",
        f"elapsed_s={elapsed_seconds:.2f},best_cost={best_cost:.6f},",
        f"current_cost={current_cost:.6f},iterations={iterations},",
        f"algorithm=gvns_ql",
        flush=True,
    )


def _temperature_scale(solution: Solution) -> float:
    """CVRP adaptation of paper's flowshop temperature normalization.

    The paper uses total processing time divided by 10*n*m. For CVRP there
    are no machine processing times; use mean route arc length as the distance
    scale, so tau_P remains dimensionless and annealing responds to instance
    coordinates. At least one arc is counted for the empty-customer case.
    """
    routes = solution.routes
    total_distance = objective(solution)
    arc_count = sum(max(1, len(route) - 1) for route in routes)
    return total_distance / arc_count


def _accept_worse(delta: float, temperature: float, rng: random.Random) -> bool:
    if delta <= 0:
        return True
    return rng.random() < math.exp(-delta / max(temperature, _EPS))


def gvns_ql(cvrp: CVRPInstance, time_limit: float = 480.0,
            seed: int = 0) -> SearchResult:
    """Run QL-guided GVNS with Clarke-Wright initialization in its time limit."""
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
    q_table = {name: [0.0] * len(actions) for name, actions in _ACTIONS.items()}
    iterations = 0

    _log_best(cvrp.name, seed, time.monotonic() - start, current_cost, best_cost, iterations)

    k_max = 2
    while not _is_deadline():
        outer_actions = _random_action_indices(rng)
        outer_params = _parameters_from_indices(outer_actions)
        outer_q_actions = {
            name: index
            for name, index in outer_actions.items()
            if name != "d_max"
        }
        temperature = max(
            _EPS, outer_params.tau_p * _temperature_scale(current)
        )
        k = 1

        while k <= k_max and not _is_deadline():
            shaken = _shake(current, k, rng)
            candidate = _vnd(
                shaken,
                best,
                q_table,
                outer_params,
                rng,
            )
            candidate_cost = objective(candidate)
            delta = candidate_cost - current_cost
            iterations += 1

            if current_cost + _EPS < best_cost:
                best, best_cost = _copy_solution(current), current_cost
                _log_best(cvrp.name, seed, time.monotonic() - start, current_cost, best_cost, iterations)

            if delta < -_EPS:
                current, current_cost = candidate, candidate_cost
                k = 1
                reward = 1.0 / max(candidate_cost, _EPS)
                _q_learning_update(
                    q_table,
                    outer_q_actions,
                    reward,
                    outer_params.learning_factor,
                    outer_params.discount_factor,
                )
            else:
                k += 1
                if _accept_worse(delta, temperature, rng):
                    current, current_cost = candidate, candidate_cost






    return SearchResult(
        solution=best,
        initial_cost=initial_cost,
        cost=best_cost,
        iterations=iterations,
        elapsed_seconds=time.monotonic() - start,
        q_values={name: tuple(values) for name, values in q_table.items()},
    )

def _vnd(solution: Solution, reference: Solution, q_table: dict[str, list[float]],
         starting_params: SearchParameters,rng: random.Random) -> Solution:

    current = solution

    q = 1
    q_max = 2
    max_customers = max(1, len(_customers(current)))
    jumping_probability = starting_params.epsilon
    while q <= q_max and not _is_deadline():
        if rng.random() < jumping_probability:
            selected = _random_action_indices(rng)
        else:
            selected = _greedy_action_indices(q_table, rng)

        params = _parameters_from_indices(selected)
        jumping_probability = params.epsilon
        vnd_temperature = max(
            _EPS, params.tau_p * _temperature_scale(current)
        )
        d_max = min(params.d_max, max_customers)
        candidate = (
            _variable_ig(current, d_max, rng, reference)
            if q == 1
            else _variable_block_insertion(current, d_max, reference)
        )
        candidate_cost = objective(candidate)
        delta = candidate_cost - objective(current)

        if delta < -_EPS:
            current, q = candidate, 1
            _q_learning_update(
                q_table,
                selected,
                1.0 / max(candidate_cost, _EPS),
                params.learning_factor,
                params.discount_factor,
            )
        else:
            q += 1
            if _accept_worse(delta, vnd_temperature, rng):
                current = candidate

    return current


