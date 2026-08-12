from __future__ import annotations

from types import SimpleNamespace

import pytest

from gear_sonic.scripts import run_livox_front_sector_filter as front_filter
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


class FakeCustomMessage:
    def __init__(self) -> None:
        self.header = None
        self.timebase = 0
        self.point_num = 0
        self.lidar_id = 0
        self.rsvd = []
        self.points = []


def custom_point(
    x: float,
    y: float,
    z: float,
    *,
    offset_time: int,
    reflectivity: int,
    tag: int,
    line: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        offset_time=offset_time,
        x=x,
        y=y,
        z=z,
        reflectivity=reflectivity,
        tag=tag,
        line=line,
    )


def test_custom_message_preserves_metadata_and_every_retained_point_field() -> None:
    header = SimpleNamespace(frame_id="livox_frame", stamp=object())
    blocked = custom_point(
        2.0, 0.0, 3.0, offset_time=10, reflectivity=11, tag=12, line=1
    )
    left = custom_point(
        1.0, 2.0, 4.0, offset_time=20, reflectivity=21, tag=22, line=2
    )
    rear = custom_point(
        -1.0, 0.0, 5.0, offset_time=30, reflectivity=31, tag=32, line=3
    )
    message = FakeCustomMessage()
    message.header = header
    message.timebase = 123456
    message.point_num = 3
    message.lidar_id = 7
    message.rsvd = [8, 9, 10]
    message.points = [blocked, left, rear]

    filtered = front_filter.filter_custom_message(message, FakeCustomMessage)

    assert filtered is not message
    assert filtered.header is header
    assert filtered.timebase == 123456
    assert filtered.lidar_id == 7
    assert filtered.rsvd == [8, 9, 10]
    assert filtered.point_num == 2
    assert filtered.points == [left, rear]
    assert filtered.points[0] is left
    assert vars(filtered.points[0]) == vars(left)
    assert vars(filtered.points[1]) == vars(rear)


def test_custom_message_publishes_a_valid_empty_result() -> None:
    message = FakeCustomMessage()
    message.points = [
        custom_point(
            1.0, 0.0, -50.0, offset_time=1, reflectivity=2, tag=3, line=4
        )
    ]
    message.point_num = 1

    filtered = front_filter.filter_custom_message(message, FakeCustomMessage)

    assert filtered.points == []
    assert filtered.point_num == 0


def test_cli_defaults_to_the_shared_raw_and_filtered_topics() -> None:
    settings = front_filter.resolve_settings(
        front_filter.build_argument_parser().parse_args([])
    )

    assert settings.input_topic == "/livox/lidar_raw"
    assert settings.output_topic == "/livox/lidar"
    assert settings.sector_degrees == 90.0


@pytest.mark.parametrize("value", ["0", "180", "nan", "inf"])
def test_cli_rejects_an_invalid_sector(value: str) -> None:
    args = front_filter.build_argument_parser().parse_args(
        ["--sector-degrees", value]
    )

    with pytest.raises(ValueError, match="sector_degrees"):
        front_filter.resolve_settings(args)


def test_cli_rejects_equal_or_relative_topics() -> None:
    parser = front_filter.build_argument_parser()
    with pytest.raises(ValueError, match="different"):
        front_filter.resolve_settings(
            parser.parse_args(
                ["--input-topic", "/livox/lidar", "--output-topic", "/livox/lidar"]
            )
        )
    with pytest.raises(ValueError, match="absolute ROS topic"):
        front_filter.resolve_settings(
            parser.parse_args(["--input-topic", "livox/lidar_raw"])
        )
