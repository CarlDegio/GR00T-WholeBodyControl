import sys
import types

import pytest

from gear_sonic.camera.gemini_server_launcher import (
    build_server_argv,
    discover_single_gemini,
    main,
    read_gemini_usb_speed,
    validate_usb3_speed,
)


class FakeDeviceInfo:
    def __init__(self, serial: str):
        self._serial = serial

    def get_serial_number(self):
        return self._serial

    def get_vid(self):
        return 0x2BC5

    def get_pid(self):
        return 0x0813


class FakeDevice:
    def __init__(self, serial: str):
        self._info = FakeDeviceInfo(serial)

    def get_device_info(self):
        return self._info


class FakeDeviceList:
    def __init__(self, serials: list[str]):
        self._devices = [FakeDevice(serial) for serial in serials]

    def get_count(self):
        return len(self._devices)

    def get_device_by_index(self, index: int):
        return self._devices[index]


class FakeContext:
    def __init__(self, serials: list[str]):
        self._devices = FakeDeviceList(serials)

    def query_devices(self):
        return self._devices


def test_discover_single_gemini_returns_device_info_serial():
    """Catch use of the SDK device-list serial cache, which can be empty."""
    assert discover_single_gemini(FakeContext(["CPMD464001G"])) == "CPMD464001G"


def test_discover_single_gemini_rejects_another_orbbec_model():
    """Catch launching a non-345Lg Orbbec device as the head camera."""
    context = FakeContext(["OTHER"])
    context._devices._devices[0]._info.get_pid = lambda: 0x080B

    with pytest.raises(RuntimeError, match="Gemini 345Lg"):
        discover_single_gemini(context)


@pytest.mark.parametrize("serials", [[], ["first", "second"]])
def test_discover_single_gemini_rejects_ambiguous_device_counts(serials):
    """Catch silently selecting no device or the first of multiple devices."""
    with pytest.raises(RuntimeError, match="exactly one"):
        discover_single_gemini(FakeContext(serials))


def test_read_gemini_usb_speed_matches_vid_pid_and_serial(tmp_path):
    """Catch reading the speed of a different USB camera."""
    wrong = tmp_path / "4-3"
    wrong.mkdir()
    (wrong / "idVendor").write_text("8086\n")
    (wrong / "idProduct").write_text("0813\n")
    (wrong / "serial").write_text("CPMD464001G\n")
    (wrong / "speed").write_text("480\n")

    gemini = tmp_path / "4-4"
    gemini.mkdir()
    (gemini / "idVendor").write_text("2bc5\n")
    (gemini / "idProduct").write_text("0813\n")
    (gemini / "serial").write_text("CPMD464001G\n")
    (gemini / "speed").write_text("5000\n")

    assert read_gemini_usb_speed("CPMD464001G", tmp_path) == 5000


def test_read_gemini_usb_speed_rejects_missing_device(tmp_path):
    """Catch continuing when the selected serial is absent from sysfs."""
    with pytest.raises(RuntimeError, match="CPMD464001G"):
        read_gemini_usb_speed("CPMD464001G", tmp_path)


def test_validate_usb3_speed_rejects_usb2():
    """Catch RGB-D launch on a 480 Mb/s USB 2 connection."""
    with pytest.raises(RuntimeError, match=r"USB 3.*480"):
        validate_usb3_speed(480)


def test_build_server_argv_starts_only_ego_view_orbbec():
    """Catch accidentally retaining the RealSense chest or wrist cameras."""
    assert build_server_argv("CPMD464001G") == [
        sys.executable,
        "-m",
        "gear_sonic.camera.composed_camera",
        "--ego-view-camera",
        "orbbec",
        "--ego-view-device-id",
        "CPMD464001G",
        "--orbbec-enable-depth",
        "--jpeg-quality",
        "95",
        "--port",
        "5555",
    ]


def test_main_execs_composed_server_for_detected_usb3_gemini(tmp_path, monkeypatch):
    """Catch preflight succeeding without entering the composed-camera server."""
    gemini = tmp_path / "4-4"
    gemini.mkdir()
    (gemini / "idVendor").write_text("2bc5\n")
    (gemini / "idProduct").write_text("0813\n")
    (gemini / "serial").write_text("CPMD464001G\n")
    (gemini / "speed").write_text("5000\n")

    fake_sdk = types.ModuleType("pyorbbecsdk")
    fake_sdk.Context = lambda: FakeContext(["CPMD464001G"])
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", fake_sdk)

    exec_call = {}

    def fake_execv(path, argv):
        exec_call.update(path=path, argv=argv)

    monkeypatch.setattr("os.execv", fake_execv)

    main(sysfs_root=tmp_path)

    assert exec_call == {
        "path": sys.executable,
        "argv": build_server_argv("CPMD464001G"),
    }
