from __future__ import annotations

from types import SimpleNamespace

from gear_sonic.scripts.run_livox_front_sector_filter import (
    is_inside_forward_sector,
    retain_points_outside_forward_sector,
)


def point(x: float, y: float, z: float = 0.0, *, label: str = "") -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, label=label)


def test_production_sector_removes_forward_points_and_closed_boundaries() -> None:
    assert is_inside_forward_sector(point(1.0, 0.0))
    assert is_inside_forward_sector(point(1.0, 1.0))
    assert is_inside_forward_sector(point(1.0, -1.0))
    assert is_inside_forward_sector(point(0.5, 0.25, 100.0))


def test_production_sector_keeps_origin_rear_and_points_outside_boundaries() -> None:
    assert not is_inside_forward_sector(point(0.0, 0.0))
    assert not is_inside_forward_sector(point(-1.0, 0.0))
    assert not is_inside_forward_sector(point(1.0, 1.0001))
    assert not is_inside_forward_sector(point(1.0, -1.0001))


def test_retained_points_keep_input_order_and_identity() -> None:
    left = point(1.0, 2.0, label="left")
    blocked = point(2.0, 0.0, label="blocked")
    rear = point(-1.0, 0.0, label="rear")

    retained = retain_points_outside_forward_sector([left, blocked, rear])

    assert retained == [left, rear]
    assert retained[0] is left
    assert retained[1] is rear


def test_sector_argument_supports_isolated_non_production_checks() -> None:
    candidate = point(1.0, 0.5)

    assert is_inside_forward_sector(candidate, sector_degrees=90.0)
    assert not is_inside_forward_sector(candidate, sector_degrees=30.0)
