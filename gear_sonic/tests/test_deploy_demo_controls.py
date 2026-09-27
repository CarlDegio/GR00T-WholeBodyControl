from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


DEPLOY_DIR = Path(__file__).resolve().parents[2] / "gear_sonic_deploy"


@pytest.mark.parametrize("address", ["127.0.0.1", "192.0.2.1"])
@pytest.mark.parametrize("flag_position", ["absent", "before", "after"])
def test_deploy_script_only_forwards_explicit_demo_flag(
    tmp_path: Path, address: str, flag_position: str,
) -> None:
    # Run a copy with a stub runner: no builds, robot processes or network access.
    script = tmp_path / "deploy.sh"
    shutil.copyfile(DEPLOY_DIR / "deploy.sh", script)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "setup_env.sh").write_text(":\n", encoding="utf-8")
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    for name in ("cmake", "clang", "git"):
        stub = mock_bin / name
        stub.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
        stub.chmod(0o755)
    runner = mock_bin / "just"
    runner.write_text(
        '#!/bin/sh\nif [ "$1" = run ]; then\n'
        '  printf "%s\\n" "$@" > "$DEMO_TEST_ARGS"\nfi\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    args_path = tmp_path / "runner_args.txt"
    env = dict(os.environ)
    env.update(
        PATH=f"{mock_bin}{os.pathsep}{env.get('PATH', '')}",
        TensorRT_ROOT=str(tmp_path),
        DEMO_TEST_ARGS=str(args_path),
    )
    args = ["--yes", "--input-type", "zmq_manager"]
    if flag_position == "before":
        args.append("--demo-controls")
    args.append(address)
    if flag_position == "after":
        args.append("--demo-controls")

    subprocess.run(
        ["bash", str(script), *args], env=env, check=True,
        capture_output=True, text=True, timeout=10,
    )

    forwarded = args_path.read_text(encoding="utf-8").splitlines()
    assert forwarded[:3] == ["run", "g1_deploy_onnx_ref", address]
    assert forwarded.count("--demo-controls") == (flag_position != "absent")
    assert ("--disable-crc-check" in forwarded) is (address == "127.0.0.1")


@pytest.mark.parametrize("name", ["planner_upper_body_hold", "manual_gripper_latch"])
def test_cpp_demo_control_isolation(tmp_path: Path, name: str) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed for the standalone C++ control tests")
    project = DEPLOY_DIR / "src/g1/g1_deploy_onnx_ref"
    binary = tmp_path / name
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic",
         "-I", str(project / "include"), str(project / "tests" / f"test_{name}.cpp"),
         "-o", str(binary)],
        capture_output=True, text=True, timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)
