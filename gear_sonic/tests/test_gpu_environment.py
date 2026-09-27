from pathlib import Path

from gear_sonic.scripts.gpu_environment import gpu_environment


def test_matching_cache_precedes_existing_libraries(tmp_path):
    repo = tmp_path / "project"
    info = tmp_path / "driver"
    info.write_text("NVIDIA UNIX x86_64 Kernel Module  580.173.02  date")
    cache = tmp_path / ".tools/nvidia-580.173.02/usr/lib/x86_64-linux-gnu"
    cache.mkdir(parents=True)
    for name in ("libcuda.so.1", "libnvidia-ml.so.1"):
        (cache / name).touch()
    assert gpu_environment(repo, environ={"LD_LIBRARY_PATH": "/old:/old:"}, driver_info=info) == (
        f"LD_LIBRARY_PATH={cache}:/old",
    )


def test_incomplete_or_mismatched_cache_is_not_selected(tmp_path):
    info = tmp_path / "driver"
    info.write_text("Kernel Module  580.178.04")
    cache = tmp_path / ".tools/nvidia-580.173.02/usr/lib/x86_64-linux-gnu"
    cache.mkdir(parents=True)
    (cache / "libcuda.so.1").touch()
    assert gpu_environment(tmp_path / "repo", environ={"LD_LIBRARY_PATH": "/existing"}, driver_info=info) == (
        "LD_LIBRARY_PATH=/existing",
    )


def test_no_driver_or_environment_needs_no_override(tmp_path):
    assert gpu_environment(tmp_path, environ={}, driver_info=tmp_path / "missing") == ()
