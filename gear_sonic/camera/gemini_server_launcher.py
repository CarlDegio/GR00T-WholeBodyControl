"""Preflight and launch one Gemini 345Lg through the composed camera server."""

import os
from pathlib import Path
import sys
from typing import Any

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY

ORBBEC_VENDOR_ID = 0x2BC5
GEMINI_345LG_PRODUCT_ID = 0x0813
MINIMUM_USB3_SPEED_MBPS = 5000


def discover_single_gemini(context: Any) -> str:
    """Return the serial of the only connected Gemini 345Lg."""
    devices = context.query_devices()
    count = devices.get_count()
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one connected Orbbec device, found {count}"
        )

    try:
        device = devices.get_device_by_index(0)
        info = device.get_device_info()
    except Exception as exc:
        raise RuntimeError(
            "Found an Orbbec device but could not open it; install the official "
            "udev rules and reconnect the camera"
        ) from exc

    if (
        info.get_vid() != ORBBEC_VENDOR_ID
        or info.get_pid() != GEMINI_345LG_PRODUCT_ID
    ):
        raise RuntimeError(
            "The connected Orbbec device is not a Gemini 345Lg "
            f"({info.get_vid():04x}:{info.get_pid():04x})"
        )

    serial = info.get_serial_number()
    if not serial:
        raise RuntimeError("Gemini 345Lg returned an empty serial number")
    return serial


def read_gemini_usb_speed(serial: str, sysfs_root: Path) -> int:
    """Read the negotiated USB speed for the exact Gemini serial from sysfs."""
    for device_path in sysfs_root.iterdir():
        if not device_path.is_dir():
            continue
        try:
            vendor = int((device_path / "idVendor").read_text().strip(), 16)
            product = int((device_path / "idProduct").read_text().strip(), 16)
            device_serial = (device_path / "serial").read_text().strip()
        except (FileNotFoundError, PermissionError, ValueError):
            continue

        if (
            vendor == ORBBEC_VENDOR_ID
            and product == GEMINI_345LG_PRODUCT_ID
            and device_serial == serial
        ):
            try:
                return int((device_path / "speed").read_text().strip())
            except (FileNotFoundError, PermissionError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not read USB speed for Gemini 345Lg {serial}"
                ) from exc

    raise RuntimeError(f"Gemini 345Lg {serial} was not found in {sysfs_root}")


def validate_usb3_speed(speed_mbps: int) -> None:
    """Reject links slower than USB 3 SuperSpeed."""
    if speed_mbps < MINIMUM_USB3_SPEED_MBPS:
        raise RuntimeError(
            "Gemini 345Lg requires a USB 3 link for RGB-D streaming; "
            f"the negotiated speed is {speed_mbps} Mb/s"
        )


def build_server_argv(serial: str) -> list[str]:
    """Build the composed-camera command for one ego-view Gemini."""
    return [
        sys.executable,
        "-m",
        "gear_sonic.camera.composed_camera",
        "--ego-view-camera",
        "orbbec",
        "--ego-view-device-id",
        serial,
        "--orbbec-enable-depth",
        "--jpeg-quality",
        str(PRODUCTION_JPEG_QUALITY),
        "--port",
        "5555",
    ]


def main(sysfs_root: Path = Path("/sys/bus/usb/devices")) -> None:
    """Validate the local Gemini and replace this process with the server."""
    from pyorbbecsdk import Context

    serial = discover_single_gemini(Context())
    speed = read_gemini_usb_speed(serial, sysfs_root)
    validate_usb3_speed(speed)
    argv = build_server_argv(serial)
    print(f"Starting Gemini {serial} at {speed} Mb/s", flush=True)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
