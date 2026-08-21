from __future__ import annotations

import json

import pytest

from gear_sonic.utils.pico_video.usb_network import (
    PicoUsbNetwork,
    PicoUsbNetworkError,
    discover_pico_usb_network,
)


class FakeCommandRunner:
    def __init__(
        self, responses: dict[tuple[str, ...], str | Exception]
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> str:
        self.calls.append(command)
        response = self.responses[command]
        if isinstance(response, Exception):
            raise response
        return response


def _addresses(*interfaces: tuple[str, str]) -> str:
    return json.dumps(
        [
            {
                "ifname": interface,
                "addr_info": [
                    {
                        "family": "inet",
                        "scope": "global",
                        "local": address,
                        "prefixlen": 24,
                    }
                ],
            }
            for interface, address in interfaces
        ]
    )


def _properties(**values: str) -> str:
    return "".join(f"{key}={value}\n" for key, value in values.items())


@pytest.mark.parametrize("source_field", ["from", "prefsrc"])
def test_discovers_only_the_active_pico_rndis_route(source_field: str) -> None:
    runner = FakeCommandRunner(
        {
            ("ip", "-j", "-4", "address", "show", "up"): _addresses(
                ("wlp131s0", "192.168.1.20"),
                ("enx4662be5cb0cb", "192.168.123.61"),
            ),
            (
                "udevadm",
                "info",
                "--query=property",
                "--path=/sys/class/net/wlp131s0",
            ): _properties(ID_NET_DRIVER="iwlwifi"),
            (
                "udevadm",
                "info",
                "--query=property",
                "--path=/sys/class/net/enx4662be5cb0cb",
            ): _properties(
                ID_USB_DRIVER="rndis_host",
                ID_VENDOR="Pico",
                ID_VENDOR_ID="05c6",
                ID_MODEL="PICO_4_Ultra",
                ID_MODEL_ID="9024",
                ID_SERIAL_SHORT="PA9410MGL1090624G",
            ),
            (
                "ip",
                "-j",
                "-4",
                "route",
                "show",
                "dev",
                "enx4662be5cb0cb",
            ): json.dumps(
                [
                    {
                        "dst": "default",
                        "gateway": "192.168.123.242",
                        "dev": "enx4662be5cb0cb",
                    }
                ]
            ),
            (
                "ip",
                "-j",
                "-4",
                "route",
                "get",
                "192.168.123.242",
                "from",
                "192.168.123.61",
            ): json.dumps(
                [
                    {
                        "dst": "192.168.123.242",
                        "dev": "enx4662be5cb0cb",
                        source_field: "192.168.123.61",
                    }
                ]
            ),
        }
    )

    network = discover_pico_usb_network(command_runner=runner)

    assert network == PicoUsbNetwork(
        interface="enx4662be5cb0cb",
        workstation_ip="192.168.123.61",
        pico_ip="192.168.123.242",
        prefix_length=24,
        serial="PA9410MGL1090624G",
    )


def test_rejects_a_non_pico_rndis_device() -> None:
    runner = FakeCommandRunner(
        {
            ("ip", "-j", "-4", "address", "show", "up"): _addresses(
                ("enxgeneric", "192.168.42.10")
            ),
            (
                "udevadm",
                "info",
                "--query=property",
                "--path=/sys/class/net/enxgeneric",
            ): _properties(
                ID_USB_DRIVER="rndis_host",
                ID_VENDOR="Qualcomm_Inc.",
                ID_VENDOR_ID="05c6",
                ID_MODEL="Alcatel_LTE_Modem",
                ID_MODEL_ID="9024",
            ),
        }
    )

    with pytest.raises(PicoUsbNetworkError, match="PICO USBOnly RNDIS"):
        discover_pico_usb_network(command_runner=runner)


def test_unrelated_udev_failure_does_not_hide_a_valid_pico() -> None:
    udev_failure = PicoUsbNetworkError("virtual interface disappeared")
    runner = FakeCommandRunner(
        {
            ("ip", "-j", "-4", "address", "show", "up"): _addresses(
                ("tun0", "10.8.0.2"),
                ("enxpico", "192.168.123.61"),
            ),
            (
                "udevadm",
                "info",
                "--query=property",
                "--path=/sys/class/net/tun0",
            ): udev_failure,
            (
                "udevadm",
                "info",
                "--query=property",
                "--path=/sys/class/net/enxpico",
            ): _properties(
                ID_USB_DRIVER="rndis_host",
                ID_VENDOR="Pico",
                ID_MODEL="PICO_4_Ultra",
                ID_SERIAL_SHORT="PA9410MGL1090624G",
            ),
            (
                "ip",
                "-j",
                "-4",
                "route",
                "show",
                "dev",
                "enxpico",
            ): json.dumps(
                [{"dst": "default", "gateway": "192.168.123.242", "dev": "enxpico"}]
            ),
            (
                "ip",
                "-j",
                "-4",
                "route",
                "get",
                "192.168.123.242",
                "from",
                "192.168.123.61",
            ): json.dumps(
                [
                    {
                        "dst": "192.168.123.242",
                        "dev": "enxpico",
                        "from": "192.168.123.61",
                    }
                ]
            ),
        }
    )

    network = discover_pico_usb_network(command_runner=runner)

    assert network.interface == "enxpico"


def test_rejects_a_route_that_would_leave_the_pico_interface() -> None:
    runner = FakeCommandRunner(
        {
            ("ip", "-j", "-4", "address", "show", "up"): _addresses(
                ("enxpico", "192.168.123.61")
            ),
            (
                "udevadm",
                "info",
                "--query=property",
                "--path=/sys/class/net/enxpico",
            ): _properties(
                ID_USB_DRIVER="rndis_host",
                ID_VENDOR="Pico",
                ID_MODEL="PICO_4_Ultra",
            ),
            (
                "ip",
                "-j",
                "-4",
                "route",
                "show",
                "dev",
                "enxpico",
            ): json.dumps(
                [{"dst": "default", "gateway": "192.168.123.242", "dev": "enxpico"}]
            ),
            (
                "ip",
                "-j",
                "-4",
                "route",
                "get",
                "192.168.123.242",
                "from",
                "192.168.123.61",
            ): json.dumps(
                [
                    {
                        "dst": "192.168.123.242",
                        "dev": "wlp131s0",
                        "prefsrc": "192.168.1.20",
                    }
                ]
            ),
        }
    )

    with pytest.raises(PicoUsbNetworkError, match="does not use PICO USB interface"):
        discover_pico_usb_network(command_runner=runner)
