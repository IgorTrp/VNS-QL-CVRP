'''
Finds an initial solution using a greedy algorithm or Clark Wright Savings algorithm.
'''
from CVRP.cvrp_instance_and_load import CVRPInstance, euclidean_2d_edge_distance
from CVRP.cvrp_solution import Solution


def initial_solution_greedy(current_instance: CVRPInstance) -> Solution:
    depot_id = current_instance.depot_ids[0]
    unserved = set(current_instance.customer_ids)

    # A customer whose demand exceeds capacity cannot fit in any route.
    too_large = [
        customer_id
        for customer_id in unserved
        if current_instance.demands[customer_id] > current_instance.capacity
    ]
    if too_large:
        raise ValueError(f"Customer demand exceeds vehicle capacity: {too_large}")

    routes = []

    while unserved:
        route = [0]  # 0 is the depot marker used in the solution representation
        current_node = depot_id
        remaining_capacity = current_instance.capacity

        while True:
            feasible_customers = [
                customer_id
                for customer_id in unserved
                if current_instance.demands[customer_id] <= remaining_capacity
            ]

            if not feasible_customers:
                break

            # Choose the nearest customer that still fits on this route.
            next_customer = min(
                feasible_customers,
                key=lambda customer_id: euclidean_2d_edge_distance(
                    current_instance.coordinates, current_node, customer_id
                ),
            )

            route.append(next_customer)
            unserved.remove(next_customer)
            remaining_capacity -= current_instance.demands[next_customer]
            current_node = next_customer

        route.append(0)
        routes.append(route)

    return Solution(routes)

# implemented following https://web.mit.edu/urban_or_book/www/book/chapter6/6.4.12.html
def initial_solution_cws(current_instance: CVRPInstance) -> Solution:
    depot_id = current_instance.depot_ids[0]
    customers = list(current_instance.customer_ids)

    def distance(first: int, second: int) -> int:
        return euclidean_2d_edge_distance(current_instance.coordinates, first, second)

    # Start with one depot-to-customer-to-depot route per customer.
    routes = {index: [customer] for index, customer in enumerate(customers)}
    route_loads = {
        index: current_instance.demands[customer]
        for index, customer in enumerate(customers)
    }
    customer_route = {
        customer: index for index, customer in enumerate(customers)
    }

    savings = [
        (
            distance(depot_id, first)
            + distance(depot_id, second)
            - distance(first, second),
            first,
            second,
        )
        for position, first in enumerate(customers)
        for second in customers[position + 1:]
    ]
    savings.sort(key=lambda item: item[0], reverse=True)

    for _, first, second in savings:
        first_route_id = customer_route[first]
        second_route_id = customer_route[second]
        if first_route_id == second_route_id:
            continue
        if (route_loads[first_route_id] + route_loads[second_route_id]
                > current_instance.capacity):
            continue

        first_route = routes[first_route_id].copy()
        second_route = routes[second_route_id].copy()

        # A saving can join routes only at their open ends, avoiding a branch.
        if first_route[-1] != first:
            if first_route[0] != first:
                continue
            first_route.reverse()
        if second_route[0] != second:
            if second_route[-1] != second:
                continue
            second_route.reverse()

        merged_route = first_route + second_route
        routes[first_route_id] = merged_route
        route_loads[first_route_id] += route_loads.pop(second_route_id)
        del routes[second_route_id]
        for customer in merged_route:
            customer_route[customer] = first_route_id

    return Solution([[0, *route, 0] for route in routes.values()])
