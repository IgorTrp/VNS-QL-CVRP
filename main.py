"""Run ordinary GVNS on the CVRP instances in ``FOLDER``."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from CVRP.cvrp_instance_and_load import load_cvrp_instances
from CVRP.cvrp_solution import is_valid
from gvns import gvns
from gvns_ql import gvns_ql

FOLDER = Path(__file__).resolve().parent / "data" / "instances"

NUMBER_OF_INSTANCES: int = 27
SEED_RANGE: int = 1
TIME_LIMIT: int = 10



def main() -> None:
    instances = load_cvrp_instances(FOLDER)
    if not instances:
        raise RuntimeError(f"No CVRP instances found in {FOLDER}")

    instances = load_cvrp_instances(FOLDER)
    for name, instance in instances.items():
        for seed in range(SEED_RANGE):
            result = gvns(instance, TIME_LIMIT, seed)
            if not is_valid(result.solution, instance):
                raise RuntimeError(f"GVNS produced an infeasible solution for {name}")


if __name__ == "__main__":
    main()