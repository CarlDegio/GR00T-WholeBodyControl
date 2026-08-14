"""Discover the native PICO USBOnly RNDIS link without managing it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import ipaddress
import json
import subprocess
from typing import Any


class PicoUsbNetworkError(RuntimeError):
    """The native PICO USBOnly network is absent, ambiguous, or misrouted."""


CommandRunner = Callable[[tuple[str, ...]], str]


@dataclass(frozen=True)
class PicoUsbNetwork:
    """One already-configured native PICO USBOnly IPv4 link."""

    interface: str
    workstation_ip: str
    pico_ip: str
    prefix_length: int
    serial: str = ""

    def __post_init__(self) -> None:
        if not self.interface:
            raise ValueError("PICO USB interface cannot be empty")
        try:
            workstation_ip = ipaddress.IPv4Address(self.workstation_ip)
            pico_ip = ipaddress.IPv4Address(self.pico_ip)
        except ipaddress.AddressValueError as exc:
            raise ValueError("PICO USB addresses must be IPv4") from exc
        if self.prefix_length <= 0 or self.prefix_length > 32:
            raise ValueError("PICO USB prefix length is outside the IPv4 range")
        network = ipaddress.IPv4Network(
            f"{workstation_ip}/{self.prefix_length}", strict=False
        )
        if pico_ip not in network or pico_ip == workstation_ip:
            raise ValueError("PICO and workstation USB addresses must share a subnet")


def _run_command(command: tuple[str, ...]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PicoUsbNetworkError(
            f"failed to inspect PICO USBOnly network with {' '.join(command)}: {exc}"
        ) from exc
    return result.stdout


def _parse_json(output: str, *, command: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise PicoUsbNetworkError(
            f"invalid JSON returned by {' '.join(command)}"
        ) from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise PicoUsbNetworkError(f"unexpected output from {' '.join(command)}")
    return value


def _parse_properties(output: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _is_pico_rndis(properties: dict[str, str]) -> bool:
    if properties.get("ID_USB_DRIVER") != "rndis_host":
        return False
    vendor = properties.get("ID_VENDOR", "").replace("_", " ").strip().upper()
    model = properties.get("ID_MODEL", "").replace("_", " ").strip().upper()
    serial = properties.get("ID_SERIAL", "").replace("_", " ").strip().upper()
    return vendor == "PICO" and ("PICO" in model or serial.startswith("PICO PICO"))


def _global_ipv4(record: dict[str, Any]) -> tuple[str, int] | None:
    addresses = [
        item
        for item in record.get("addr_info", [])
        if isinstance(item, dict)
        and item.get("family") == "inet"
        and item.get("scope") == "global"
        and item.get("local")
    ]
    if len(addresses) != 1:
        return None
    return str(addresses[0]["local"]), int(addresses[0]["prefixlen"])


def discover_pico_usb_network(
    interface: str | None = None,
    *,
    command_runner: CommandRunner | None = None,
) -> PicoUsbNetwork:
    """Return the sole active native PICO RNDIS network or fail closed."""

    runner = command_runner or _run_command
    address_command = ("ip", "-j", "-4", "address", "show", "up")
    records = _parse_json(runner(address_command), command=address_command)
    candidates: list[PicoUsbNetwork] = []
    unusable: list[str] = []
    inspection_failures: list[str] = []

    for record in records:
        ifname = str(record.get("ifname", ""))
        if not ifname or ifname == "lo" or (interface and ifname != interface):
            continue
        property_command = (
            "udevadm",
            "info",
            "--query=property",
            f"--path=/sys/class/net/{ifname}",
        )
        try:
            properties = _parse_properties(runner(property_command))
        except PicoUsbNetworkError as exc:
            inspection_failures.append(f"{ifname}: {exc}")
            continue
        if not _is_pico_rndis(properties):
            continue

        address = _global_ipv4(record)
        if address is None:
            unusable.append(f"{ifname} has no unique global IPv4 address")
            continue
        workstation_ip, prefix_length = address

        route_command = ("ip", "-j", "-4", "route", "show", "dev", ifname)
        routes = _parse_json(runner(route_command), command=route_command)
        gateways = {
            str(route["gateway"])
            for route in routes
            if route.get("dst") == "default" and route.get("gateway")
        }
        if len(gateways) != 1:
            unusable.append(f"{ifname} has no unique PICO USB gateway")
            continue
        pico_ip = gateways.pop()

        route_get_command = (
            "ip",
            "-j",
            "-4",
            "route",
            "get",
            pico_ip,
            "from",
            workstation_ip,
        )
        selected_routes = _parse_json(
            runner(route_get_command), command=route_get_command
        )
        if not selected_routes:
            unusable.append(f"{ifname} has no route to PICO peer {pico_ip}")
            continue
        selected = selected_routes[0]
        selected_source = selected.get("from") or selected.get("prefsrc")
        if selected.get("dev") != ifname or selected_source != workstation_ip:
            unusable.append(
                f"route to {pico_ip} does not use PICO USB interface {ifname}"
            )
            continue

        try:
            candidate = PicoUsbNetwork(
                interface=ifname,
                workstation_ip=workstation_ip,
                pico_ip=pico_ip,
                prefix_length=prefix_length,
                serial=properties.get("ID_SERIAL_SHORT", ""),
            )
        except ValueError as exc:
            unusable.append(f"{ifname}: {exc}")
            continue
        candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(candidate.interface for candidate in candidates)
        raise PicoUsbNetworkError(
            f"multiple PICO USBOnly RNDIS networks found ({names}); select one explicitly"
        )
    if unusable:
        raise PicoUsbNetworkError("; ".join(unusable))
    suffix = f" on interface {interface}" if interface else ""
    inspection_detail = (
        f"; inspection failures: {'; '.join(inspection_failures)}"
        if inspection_failures
        else ""
    )
    raise PicoUsbNetworkError(
        "PICO USBOnly RNDIS network not found"
        f"{suffix}; connect PICO, select USBOnly, and wait for DHCP"
        f"{inspection_detail}"
    )
