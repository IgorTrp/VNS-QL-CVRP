"""CVRP solution representation and feasibility validation.
Routes use ``0`` as the depot , for example::
    Solution([[0, 4, 7, 2, 0], [0, 3, 8, 5, 0], [0, 1, 6, 0]])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .cvrp_instance_and_load import CVRPInstance


DEPOT = 0


@dataclass
class Solution:
    routes: list[list[int]]

    def __post_init__(self) -> None:
        # Copy the nested lists so later edits to the caller's input do not
        # silently modify this solution.
        self.routes = [list(route) for route in self.routes]

        #printing
    def __repr__(self) -> str:
        return f"Solution(routes={self.routes!r})"


def is_valid(solution: Solution | Sequence[Sequence[int]], instance: CVRPInstance, missing_customers: set[int] | None = None) -> bool:
    routes = solution.routes if isinstance(solution, Solution) else solution

    all_customers = set(instance.customer_ids)
    missing = set(missing_customers or ())

    if not missing <= all_customers:
        return False

    expected_customers = all_customers - missing
    visited_customers: list[int] = []

    for route in routes:
        if len(route) < 2 or route[0] != DEPOT or route[-1] != DEPOT:
            return False

        customers = route[1:-1]
        if DEPOT in customers or any(customer not in expected_customers for customer in customers):
            return False

        route_demand = sum(instance.demands[customer] for customer in customers)
        if route_demand > instance.capacity:
            return False

        visited_customers.extend(customers)

    return len(visited_customers) == len(expected_customers) and set(visited_customers) == expected_customers