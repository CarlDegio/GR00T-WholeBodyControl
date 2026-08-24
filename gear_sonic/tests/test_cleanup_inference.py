from __future__ import annotations

from gear_sonic.scripts import cleanup_inference


def test_external_inference_server_is_not_an_automatic_cleanup_target() -> None:
    patterns = "\n".join(cleanup_inference.INFERENCE_PROCESS_PATTERNS)

    assert "g1_sonic_zmq_policy" not in patterns


def test_cleanup_unlinks_only_sonic_shared_memory_files(tmp_path) -> None:
    stale = tmp_path / "sonic-camera-abc123"
    unrelated = tmp_path / "unrelated"
    directory = tmp_path / "sonic-directory"
    stale.write_bytes(b"frame")
    unrelated.write_bytes(b"keep")
    directory.mkdir()

    removed = cleanup_inference.cleanup_shared_memory_files(tmp_path)

    assert removed == (stale,)
    assert not stale.exists()
    assert unrelated.read_bytes() == b"keep"
    assert directory.is_dir()


def test_full_cleanup_stops_processes_before_unlinking(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        cleanup_inference,
        "terminate_matching_processes",
        lambda patterns: events.append(
            "stop-gateway" if cleanup_inference.GATEWAY_PROCESS_PATTERNS[0] in patterns else "stop"
        ),
    )
    monkeypatch.setattr(
        cleanup_inference,
        "cleanup_shared_memory_files",
        lambda: events.append("unlink"),
    )

    cleanup_inference.cleanup_inference_processes()

    assert events == ["stop-gateway", "unlink"]
