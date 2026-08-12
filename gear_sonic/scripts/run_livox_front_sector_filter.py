#!/usr/bin/env python3
"""Filter the robot-forward sector from Livox CustomMsg point clouds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence


def _validate_sector_degrees(sector_degrees: float) -> float:
    value = float(sector_degrees)
    if not math.isfinite(value) or not 0.0 < value < 180.0:
        raise ValueError("sector_degrees must be finite and between 0 and 180")
    return value


def is_inside_forward_sector(
    point: Any,
    sector_degrees: float = 90.0,
) -> bool:
    sector = _validate_sector_degrees(sector_degrees)
    x = float(point.x)
    y = float(point.y)
    if x <= 0.0:
        return False
    if sector == 90.0:
        return abs(y) <= x
    return abs(math.degrees(math.atan2(y, x))) <= sector / 2.0


def retain_points_outside_forward_sector(
    points: Sequence[Any],
    sector_degrees: float = 90.0,
) -> list[Any]:
    sector = _validate_sector_degrees(sector_degrees)
    return [
        point
        for point in points
        if not is_inside_forward_sector(point, sector)
    ]


@dataclass(frozen=True)
class FrontSectorFilterSettings:
    input_topic: str
    output_topic: str
    sector_degrees: float


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-topic", default="/livox/lidar_raw")
    parser.add_argument("--output-topic", default="/livox/lidar")
    parser.add_argument("--sector-degrees", type=float, default=90.0)
    return parser


def resolve_settings(args: argparse.Namespace) -> FrontSectorFilterSettings:
    input_topic = str(args.input_topic)
    output_topic = str(args.output_topic)
    if not input_topic.startswith("/") or not output_topic.startswith("/"):
        raise ValueError("input_topic and output_topic must be absolute ROS topics")
    if input_topic == output_topic:
        raise ValueError("input_topic and output_topic must be different")
    return FrontSectorFilterSettings(
        input_topic=input_topic,
        output_topic=output_topic,
        sector_degrees=_validate_sector_degrees(args.sector_degrees),
    )


def filter_custom_message(
    message: Any,
    message_factory: Callable[[], Any],
    sector_degrees: float = 90.0,
) -> Any:
    retained = retain_points_outside_forward_sector(
        message.points,
        sector_degrees,
    )
    filtered = message_factory()
    filtered.header = message.header
    filtered.timebase = message.timebase
    filtered.lidar_id = message.lidar_id
    filtered.rsvd = message.rsvd
    filtered.points = retained
    filtered.point_num = len(retained)
    return filtered


def run_filter(settings: FrontSectorFilterSettings) -> None:
    import rclpy
    from livox_ros_driver2.msg import CustomMsg

    rclpy.init(args=None)
    node = rclpy.create_node("sonic_livox_front_sector_filter")
    publisher = node.create_publisher(CustomMsg, settings.output_topic, 10)

    def on_message(message: Any) -> None:
        publisher.publish(
            filter_custom_message(
                message,
                CustomMsg,
                settings.sector_degrees,
            )
        )

    subscription = node.create_subscription(
        CustomMsg,
        settings.input_topic,
        on_message,
        10,
    )
    node.get_logger().info(
        "Filtering %.1f degrees from %s to %s"
        % (
            settings.sector_degrees,
            settings.input_topic,
            settings.output_topic,
        )
    )
    try:
        rclpy.spin(node)
    finally:
        del subscription
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    settings = resolve_settings(build_argument_parser().parse_args())
    run_filter(settings)


if __name__ == "__main__":
    main()
