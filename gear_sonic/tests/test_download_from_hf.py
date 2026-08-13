from __future__ import annotations

import sys
from pathlib import Path

import pytest

import download_from_hf


def test_sonic_v1_1_deploy_files_are_downloaded_as_a_matched_set() -> None:
    assert download_from_hf.SONIC_V1_1_POLICY_FILES == [
        (
            "sonic_v1_1/model_encoder.onnx",
            "policy/sonic_v1_1/model_encoder.onnx",
        ),
        (
            "sonic_v1_1/model_decoder.onnx",
            "policy/sonic_v1_1/model_decoder.onnx",
        ),
        (
            "sonic_v1_1/observation_config.yaml",
            "policy/sonic_v1_1/observation_config.yaml",
        ),
    ]


def test_low_latency_deploy_files_are_downloaded_as_a_matched_set() -> None:
    assert download_from_hf.LOW_LATENCY_POLICY_FILES == [
        (
            "low_latency/model_encoder.onnx",
            "policy/low_latency/model_encoder.onnx",
        ),
        (
            "low_latency/model_decoder.onnx",
            "policy/low_latency/model_decoder.onnx",
        ),
        (
            "low_latency/observation_config.yaml",
            "policy/low_latency/observation_config.yaml",
        ),
    ]


def test_model_variant_cli_flags_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_from_hf.py", "--low-latency", "--sonic-v1-1"],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_from_hf.parse_args()

    assert exc_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "attribute"),
    [
        ("--low-latency", "low_latency"),
        ("--sonic-v1-1", "sonic_v1_1"),
    ],
)
def test_model_variant_cli_flag_selects_requested_variant(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    attribute: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["download_from_hf.py", flag])

    args = download_from_hf.parse_args()

    assert getattr(args, attribute) is True


@pytest.mark.parametrize(
    ("flag", "expected_files"),
    [
        ("--low-latency", download_from_hf.LOW_LATENCY_POLICY_FILES),
        ("--sonic-v1-1", download_from_hf.SONIC_V1_1_POLICY_FILES),
    ],
)
def test_main_dispatches_selected_deployment_variant_to_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    expected_files: list[tuple[str, str]],
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_from_hf.py",
            flag,
            "--no-planner",
            "--output-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        download_from_hf,
        "_ensure_huggingface_hub",
        lambda: (object(), object()),
    )

    def record_download(
        _client: object,
        repo_id: str,
        hf_filename: str,
        local_dest: Path,
        token: str | None = None,
    ) -> None:
        assert repo_id == download_from_hf.REPO_ID
        assert token is None
        calls.append((hf_filename, local_dest))

    monkeypatch.setattr(download_from_hf, "download_file", record_download)

    download_from_hf.main()

    assert calls == [
        (hf_filename, tmp_path / local_rel)
        for hf_filename, local_rel in expected_files
    ]
