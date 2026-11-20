"""Data structures and loaders for TSPLIB-style CVRP instances.

Example:
    instance = load_cvrp_instance("A-n32-k5.vrp")
    instances = load_cvrp_instances(r"D:\\instances\\A-VRP")
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, hypot
from pathlib import Path
from typing import Mapping


Coordinate = tuple[float, float]


def euclidean_2d_edge_distance(
    coordinates: Mapping[int, Coordinate], first: int, second: int
) -> int:
    """Return TSPLIB EUC_2D distance rounded to the nearest integer."""
    first_x, first_y = coordinates[first]
    second_x, second_y = coordinates[second]
    return floor(hypot(first_x - second_x, first_y - second_y) + 0.5)


@dataclass(frozen=True, slots=True)
class CVRPInstance:
    """One capacitated vehicle-routing problem instance.

    Node identifiers are the values used in the source file.  ``coordinates``
    and ``demands`` therefore remain valid even when identifiers are not
    consecutive integers.  Depots normally have zero demand.
    """

    name: str
    capacity: int
    coordinates: Mapping[int, Coordinate]
    demands: Mapping[int, int]
    depot_ids: tuple[int, ...]
    edge_weight_type: str = "EUC_2D"
    comment: str | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        coordinate_ids = set(self.coordinates)
        demand_ids = set(self.demands)

        #Error checks
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not coordinate_ids:
            raise ValueError("an instance must contain at least one node")
        if coordinate_ids != demand_ids:
            missing_demands = sorted(coordinate_ids - demand_ids)
            missing_coordinates = sorted(demand_ids - coordinate_ids)
            raise ValueError(
                "coordinates and demands must cover the same nodes "
                f"(missing demands={missing_demands}, "
                f"missing coordinates={missing_coordinates})"
            )
        if not self.depot_ids:
            raise ValueError("an instance must define at least one depot")
        if not set(self.depot_ids).issubset(coordinate_ids):
            raise ValueError("every depot must have coordinates")
        if any(demand < 0 for demand in self.demands.values()):
            raise ValueError("demands cannot be negative")

    @property
    def node_ids(self) -> tuple[int, ...]:
        """All node IDs in ascending order."""
        return tuple(sorted(self.coordinates))

    @property
    def customer_ids(self) -> tuple[int, ...]:
        """All non-depot node IDs in ascending order."""
        depots = set(self.depot_ids)
        return tuple(node_id for node_id in self.node_ids if node_id not in depots)

    @property
    def dimension(self) -> int:
        """Total number of nodes, including depots."""
        return len(self.coordinates)


def load_cvrp_instance(path: str | Path) -> CVRPInstance:
    """Load one TSPLIB-style CVRP ``.vrp`` file.

    Supports the coordinate, demand, and depot sections used by the Augerat
    A-VRP instances.  It raises ``ValueError`` with the source path when a
    required field or section is missing.
    """
    source_path = Path(path)
    headers: dict[str, str] = {}
    coordinates: dict[int, Coordinate] = {}
    demands: dict[int, int] = {}
    depot_ids: list[int] = []
    section: str | None = None

    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = source_path.read_text(encoding="latin-1").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "EOF":
            continue

        if line in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}:
            section = line
            continue

        if section == "NODE_COORD_SECTION":
            fields = line.split()
            if len(fields) != 3:
                raise _parse_error(source_path, f"invalid coordinate row: {raw_line!r}")
            node_id = _parse_int(fields[0], source_path, "node ID")
            coordinates[node_id] = (float(fields[1]), float(fields[2]))
            continue

        if section == "DEMAND_SECTION":
            fields = line.split()
            if len(fields) != 2:
                raise _parse_error(source_path, f"invalid demand row: {raw_line!r}")
            node_id = _parse_int(fields[0], source_path, "node ID")
            demands[node_id] = _parse_int(fields[1], source_path, "demand")
            continue

        if section == "DEPOT_SECTION":
            depot_id = _parse_int(line.split()[0], source_path, "depot ID")
            if depot_id != -1:
                depot_ids.append(depot_id)
            continue

        if ":" in line:
            key, value = line.split(":", maxsplit=1)
            headers[key.strip().upper()] = value.strip()

    _require(headers, "NAME", source_path)
    _require(headers, "CAPACITY", source_path)
    _require_section(coordinates, "NODE_COORD_SECTION", source_path)
    _require_section(demands, "DEMAND_SECTION", source_path)
    _require_section(depot_ids, "DEPOT_SECTION", source_path)

    declared_dimension = headers.get("DIMENSION")
    if declared_dimension is not None:
        expected = _parse_int(declared_dimension, source_path, "DIMENSION")
        if expected != len(coordinates):
            raise _parse_error(
                source_path,
                f"DIMENSION is {expected}, but {len(coordinates)} coordinates were found",
            )

    return CVRPInstance(
        name=headers["NAME"],
        capacity=_parse_int(headers["CAPACITY"], source_path, "CAPACITY"),
        coordinates=coordinates,
        demands=demands,
        depot_ids=tuple(depot_ids),
        edge_weight_type=headers.get("EDGE_WEIGHT_TYPE", "EUC_2D"),
        comment=headers.get("COMMENT"),
        source_path=source_path,
    )


def load_cvrp_instances(folder: str | Path, pattern: str = "*.vrp") -> dict[str, CVRPInstance]:
    """Load every matching CVRP instance in ``folder``, keyed by instance name.

    Files are processed in name order.  A duplicate TSPLIB ``NAME`` is rejected
    so that no loaded instance is silently overwritten.
    """
    directory = Path(folder)
    if not directory.is_dir():
        raise NotADirectoryError(f"CVRP instance folder does not exist: {directory}")

    instances: dict[str, CVRPInstance] = {}
    for file_path in sorted(directory.glob(pattern)):
        instance = load_cvrp_instance(file_path)
        if instance.name in instances:
            raise ValueError(f"duplicate instance name {instance.name!r} in {directory}")
        instances[instance.name] = instance
    return instances


def _parse_int(value: str, source_path: Path, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise _parse_error(source_path, f"invalid {field_name}: {value!r}") from error


def _require(headers: Mapping[str, str], key: str, source_path: Path) -> None:
    if key not in headers:
        raise _parse_error(source_path, f"missing required header {key}")


def _require_section(values: object, section_name: str, source_path: Path) -> None:
    if not values:
        raise _parse_error(source_path, f"missing or empty {section_name}")


def _parse_error(source_path: Path, message: str) -> ValueError:
    return ValueError(f"{source_path}: {message}")
