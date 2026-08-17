import sys
import types

import gear_sonic.camera.composed_camera as composed_camera
from gear_sonic.camera.composed_camera import (
    ComposedCameraConfig,
    ComposedCameraSensor,
)


def test_orbbec_factory_propagates_fps_depth_mount_and_serial(monkeypatch):
    """Catch a missing Orbbec branch or dropped runtime configuration."""
    fake_module = types.ModuleType("gear_sonic.camera.drivers.orbbec")

    class FakeConfig:
        fps = 30
        enable_depth = True

    captured = {}

    class FakeSensor:
        def __init__(self, *, config, mount_position, device_id):
            captured.update(
                config=config,
                mount_position=mount_position,
                device_id=device_id,
            )

    fake_module.OrbbecConfig = FakeConfig
    fake_module.OrbbecSensor = FakeSensor
    monkeypatch.setitem(sys.modules, "gear_sonic.camera.drivers.orbbec", fake_module)

    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(fps=10, orbbec_enable_depth=True)
    sensor = composed._instantiate_camera("ego_view", "orbbec", "CPMD464001G")

    assert isinstance(sensor, FakeSensor)
    assert captured["config"].fps == 10
    assert captured["config"].enable_depth is True
    assert captured["mount_position"] == "ego_view"
    assert captured["device_id"] == "CPMD464001G"


def test_server_entry_closes_camera_after_keyboard_interrupt(monkeypatch):
    """Catch Ctrl-C leaving the non-daemon camera worker and USB device alive."""
    calls = []

    class FakeComposedSensor:
        def __init__(self, config):
            calls.append(("init", config))

        def run_server(self):
            calls.append(("run", None))
            raise KeyboardInterrupt

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(
        composed_camera, "ComposedCameraSensor", FakeComposedSensor
    )
    config = ComposedCameraConfig()

    composed_camera.run_server_from_config(config)

    assert calls == [("init", config), ("run", None), ("close", None)]
