"""Run ordinary GVNS on the CVRP instances in ``FOLDER``."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from CVRP.cvrp_instance_and_load import load_cvrp_instances
from CVRP.cvrp_solution import is_valid
from gvns import gvns

FOLDER = Path(__file__).resolve().parent / "data" / "instances"
NUMBER_OF_INSTANCES: int = 3
SEED_RANGE: int = 5
TIME_LIMIT: int = 480
LOG_FILE = Path(__file__).resolve().parent / "data" / "logs" / "log.txt"


def main() -> None:
    instances = load_cvrp_instances(FOLDER)
    if not instances:
        raise RuntimeError(f"No CVRP instances found in {FOLDER}")

    print(f"Loaded {len(instances)} CVRP instances from {FOLDER}")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    for name, instance in islice(instances.items(), NUMBER_OF_INSTANCES):
        for seed in range(SEED_RANGE):
            result = gvns(instance, TIME_LIMIT, seed)
            if not is_valid(result.solution, instance):
                raise RuntimeError(f"GVNS produced an infeasible solution for {name}")

            with LOG_FILE.open("a", encoding="utf-8") as log_file:
                print(
                    f"{name}, seed={seed}: initial distance="
                    f"{result.initial_cost:.3f}, best distance={result.cost:.3f}, "
                    f"routes={len(result.solution.routes)}, "
                    f"iterations={result.iterations}, "
                    f"elapsed={result.elapsed_seconds:.2f}s",
                    file=log_file,
                )


if __name__ == "__main__":
    main()