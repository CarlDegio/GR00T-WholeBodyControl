from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.yoloe26m.reference_pipeline import (
    DEFAULT_MODEL,
    DetectionSummary,
    GroundingValidationError,
    NormalizedBox,
    PixelBox,
    YoloVisualPromptConfig,
    build_grounding_prompt,
    normalized_boxes_to_pixels,
    prepare_numbered_run_dir,
    run_yoloe_visual_prompt,
    validate_grounding_response,
)


def _ready_response(*boxes: list[float]) -> dict[str, object]:
    return {
        "status": "READY",
        "target": "blue basket",
        "boxes": [
            {"bbox_2d": box, "confidence": 0.9 - index * 0.1}
            for index, box in enumerate(boxes)
        ],
        "limitations": "",
    }


def test_grounding_prompt_requires_explicit_nonempty_target() -> None:
    with pytest.raises(ValueError, match="target must be non-empty"):
        build_grounding_prompt("   ")


def test_grounding_prompt_includes_operator_target() -> None:
    prompt = build_grounding_prompt("blue basket")

    assert 'exact target: "blue basket"' in prompt
    assert "every visible matching physical instance" in prompt


def test_validate_grounding_preserves_every_box_in_order() -> None:
    response = _ready_response(
        [100, 200, 300, 400],
        [500, 100, 900, 800],
    )

    boxes = validate_grounding_response(response)

    assert boxes == [
        NormalizedBox(100.0, 200.0, 300.0, 400.0, 0.9),
        NormalizedBox(500.0, 100.0, 900.0, 800.0, 0.8),
    ]


@pytest.mark.parametrize("status", ["NOT_FOUND", "UNSURE"])
def test_validate_grounding_rejects_nonready_status(status: str) -> None:
    response = {
        "status": status,
        "target": "blue basket",
        "boxes": [],
        "limitations": "not visible",
    }

    with pytest.raises(GroundingValidationError, match=status):
        validate_grounding_response(response)


def test_validate_grounding_rejects_ready_with_no_boxes() -> None:
    with pytest.raises(GroundingValidationError, match="at least one box"):
        validate_grounding_response(_ready_response())


@pytest.mark.parametrize(
    "bad_box",
    [
        [-1, 100, 200, 300],
        [100, 100, 1001, 300],
        [300, 100, 200, 300],
        [100, 300, 200, 300],
        [100, 100, float("nan"), 300],
        [100, 100, 200],
    ],
)
def test_validate_grounding_rejects_invalid_normalized_box(
    bad_box: list[float],
) -> None:
    with pytest.raises(GroundingValidationError, match="box 1"):
        validate_grounding_response(_ready_response(bad_box))


def test_normalized_box_conversion_uses_reference_image_dimensions() -> None:
    boxes = [NormalizedBox(100.0, 200.0, 800.0, 900.0, 0.75)]

    pixels = normalized_boxes_to_pixels(boxes, width=640, height=480)

    assert [box.xyxy for box in pixels] == [(64, 96, 512, 432)]
    assert pixels[0].confidence == 0.75


@pytest.mark.parametrize("width,height", [(0, 480), (640, 0), (-1, 480)])
def test_normalized_box_conversion_rejects_invalid_image_size(
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError, match="image dimensions must be positive"):
        normalized_boxes_to_pixels(
            [NormalizedBox(0.0, 0.0, 1000.0, 1000.0, 1.0)],
            width=width,
            height=height,
        )


def test_numbered_run_dir_uses_target_and_starts_at_one(tmp_path: Path) -> None:
    result = prepare_numbered_run_dir(tmp_path, "table")

    assert result == (tmp_path / "table1").resolve()
    assert result.is_dir()


def test_numbered_run_dir_uses_max_existing_number_without_filling_gaps(
    tmp_path: Path,
) -> None:
    (tmp_path / "table1").mkdir()
    (tmp_path / "table3").mkdir()

    result = prepare_numbered_run_dir(tmp_path, "table")

    assert result.name == "table4"


def test_numbered_run_dir_slugifies_spaces(tmp_path: Path) -> None:
    result = prepare_numbered_run_dir(tmp_path, "blue  basket")

    assert result.name == "blue_basket1"


def test_numbered_run_dir_preserves_unicode_letters(tmp_path: Path) -> None:
    result = prepare_numbered_run_dir(tmp_path, "蓝色篮子")

    assert result.name == "蓝色篮子1"


def test_numbered_run_dir_ignores_unrelated_entries(tmp_path: Path) -> None:
    (tmp_path / "table2.txt").write_text("not a directory", encoding="utf-8")
    (tmp_path / "table-old").mkdir()
    (tmp_path / "other9").mkdir()

    result = prepare_numbered_run_dir(tmp_path, "table")

    assert result.name == "table1"


def test_numbered_run_dir_rejects_empty_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target must be non-empty"):
        prepare_numbered_run_dir(tmp_path, "   ")


class _FakeTensor:
    def __init__(self, values: object):
        self._values = values

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> object:
        return self._values


class _FakeBoxes:
    xyxy = _FakeTensor([[10.0, 20.0, 110.0, 220.0], [300.0, 50.0, 500.0, 350.0]])
    conf = _FakeTensor([0.91, 0.73])
    cls = _FakeTensor([0.0, 0.0])

    def __len__(self) -> int:
        return 2


class _FakeMasks:
    data = _FakeTensor(np.zeros((2, 8, 8), dtype=np.float32))

    def __len__(self) -> int:
        return 2


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes()
        self.masks = _FakeMasks()
        self.names: dict[int, str] = {0: "object0"}

    def plot(self) -> np.ndarray:
        return np.full((24, 32, 3), 127, dtype=np.uint8)


class _FakeYoloModel:
    def __init__(self, model_path: str, calls: list[dict[str, object]]) -> None:
        self.model_path = model_path
        self.calls = calls

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        self.calls.append(kwargs)
        return [_FakeResult()]


def _write_test_image(path: Path, *, width: int = 640, height: int = 480) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_yolo_runner_passes_all_boxes_as_visual_class_zero_and_saves_artifacts(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.jpg"
    target = tmp_path / "target.jpg"
    model_path = tmp_path / "model.pt"
    _write_test_image(reference)
    _write_test_image(target)
    model_path.write_bytes(b"fake-model")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[dict[str, object]] = []
    config = YoloVisualPromptConfig(
        source_image=target,
        reference_image=reference,
        target_name="blue basket",
        model_path=model_path,
        run_dir=run_dir,
        device="cpu",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        half=False,
    )
    prompt_boxes = [
        PixelBox(64, 96, 192, 240, 0.9),
        PixelBox(320, 48, 576, 384, 0.8),
    ]

    summary = run_yoloe_visual_prompt(
        config,
        prompt_boxes,
        model_factory=lambda path: _FakeYoloModel(path, calls),
        predictor_cls=object,
    )

    assert isinstance(summary, DetectionSummary)
    assert summary.detection_count == 2
    assert summary.mask_count == 2
    assert len(calls) == 1
    visual_prompts = calls[0]["visual_prompts"]
    assert isinstance(visual_prompts, dict)
    np.testing.assert_array_equal(
        visual_prompts["bboxes"],
        np.array([[64, 96, 192, 240], [320, 48, 576, 384]], dtype=np.float32),
    )
    np.testing.assert_array_equal(visual_prompts["cls"], np.array([0, 0], dtype=np.int32))
    assert calls[0]["refer_image"] == str(reference.resolve())
    assert calls[0]["source"] == str(target.resolve())
    assert calls[0]["save"] is False
    assert summary.reference_annotated.is_file()
    assert summary.target_annotated.is_file()
    assert summary.detections_json.is_file()
    payload = json.loads(summary.detections_json.read_text(encoding="utf-8"))
    assert payload["target"] == "blue basket"
    assert payload["reference_boxes_xyxy"] == [
        [64, 96, 192, 240],
        [320, 48, 576, 384],
    ]
    assert [item["bbox_xyxy"] for item in payload["detections"]] == [
        [10.0, 20.0, 110.0, 220.0],
        [300.0, 50.0, 500.0, 350.0],
    ]


def test_default_model_path_targets_installed_yoloe_26m_weight() -> None:
    assert DEFAULT_MODEL.name == "yoloe-26m-seg.pt"


class _FakeGroundingClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.response


def test_auto_pipeline_sends_only_reference_image_and_hands_every_box_to_yolo(
    tmp_path: Path,
) -> None:
    from tools.yoloe26m.reference_pipeline import run_auto_reference_pipeline

    reference = tmp_path / "reference.jpg"
    target = tmp_path / "target.jpg"
    model_path = tmp_path / "model.pt"
    _write_test_image(reference, width=640, height=480)
    _write_test_image(target, width=640, height=480)
    model_path.write_bytes(b"fake-model")
    run_dir = tmp_path / "auto"
    run_dir.mkdir()
    config = YoloVisualPromptConfig(
        source_image=target,
        reference_image=reference,
        target_name="blue basket",
        model_path=model_path,
        run_dir=run_dir,
        device="cpu",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        half=False,
    )
    client = _FakeGroundingClient(
        _ready_response([100, 200, 300, 500], [500, 100, 900, 800])
    )
    received: list[tuple[YoloVisualPromptConfig, list[PixelBox]]] = []

    def fake_yolo_runner(
        passed_config: YoloVisualPromptConfig,
        boxes: list[PixelBox],
    ) -> DetectionSummary:
        received.append((passed_config, boxes))
        target_render = run_dir / "target_annotated.jpg"
        target_render.write_bytes(b"render")
        detections = run_dir / "detections.json"
        detections.write_text("{}\n", encoding="utf-8")
        reference_render = run_dir / "reference_annotated.jpg"
        reference_render.write_bytes(b"reference")
        return DetectionSummary(
            run_dir=run_dir,
            reference_annotated=reference_render,
            target_annotated=target_render,
            detections_json=detections,
            detection_count=2,
            mask_count=2,
        )

    summary = run_auto_reference_pipeline(
        config,
        backend="codex",
        client=client,
        yolo_runner=fake_yolo_runner,
    )

    assert summary.detection_count == 2
    assert len(client.calls) == 1
    assert client.calls[0]["image_paths"] == [reference.resolve()]
    assert client.calls[0]["cwd"] == run_dir
    assert 'exact target: "blue basket"' in str(client.calls[0]["prompt"])
    assert len(received) == 1
    assert received[0][0] is config
    assert [box.xyxy for box in received[0][1]] == [
        (64, 96, 192, 240),
        (320, 48, 576, 384),
    ]
    assert (run_dir / "grounding_prompt.txt").is_file()
    assert (run_dir / "grounding_schema.json").is_file()
    assert json.loads(
        (run_dir / "grounding_response.json").read_text(encoding="utf-8")
    )["boxes"] == [
        {"bbox_2d": [100, 200, 300, 500], "confidence": 0.9},
        {"bbox_2d": [500, 100, 900, 800], "confidence": 0.8},
    ]


def test_auto_pipeline_does_not_invoke_yolo_for_empty_grounding(
    tmp_path: Path,
) -> None:
    from tools.yoloe26m.reference_pipeline import run_auto_reference_pipeline

    reference = tmp_path / "reference.jpg"
    target = tmp_path / "target.jpg"
    model_path = tmp_path / "model.pt"
    _write_test_image(reference)
    _write_test_image(target)
    model_path.write_bytes(b"fake-model")
    run_dir = tmp_path / "auto-empty"
    run_dir.mkdir()
    config = YoloVisualPromptConfig(
        source_image=target,
        reference_image=reference,
        target_name="blue basket",
        model_path=model_path,
        run_dir=run_dir,
        device="cpu",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        half=False,
    )
    client = _FakeGroundingClient(
        {
            "status": "NOT_FOUND",
            "target": "blue basket",
            "boxes": [],
            "limitations": "not visible",
        }
    )
    yolo_called = False

    def forbidden_yolo_runner(
        passed_config: YoloVisualPromptConfig,
        boxes: list[PixelBox],
    ) -> DetectionSummary:
        nonlocal yolo_called
        yolo_called = True
        raise AssertionError("YOLO must not run without a valid reference box")

    with pytest.raises(GroundingValidationError, match="NOT_FOUND"):
        run_auto_reference_pipeline(
            config,
            backend="qwenvl",
            client=client,
            yolo_runner=forbidden_yolo_runner,
        )

    assert yolo_called is False
    assert (run_dir / "grounding_response.json").is_file()


def test_create_codex_client_uses_sol_xhigh_base_pose_client() -> None:
    from gear_sonic.utils.inference.base_pose import CodexStructuredVisionClient
    from tools.yoloe26m.reference_pipeline import create_grounding_client

    client = create_grounding_client("codex")

    assert isinstance(client, CodexStructuredVisionClient)
    assert client.model == "gpt-5.6-sol"
    assert client.reasoning_effort == "xhigh"
    assert client.fast is True
    assert client.timeout_seconds == 600.0


def test_create_qwenvl_client_reuses_base_pose_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.utils.inference.base_pose import (
        DEFAULT_QWENVL_BASE_URL,
        DEFAULT_QWENVL_PLUS_MODEL,
        QwenVLStructuredVisionClient,
    )
    from tools.yoloe26m.reference_pipeline import create_grounding_client

    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-only")
    client = create_grounding_client("qwenvl")

    assert isinstance(client, QwenVLStructuredVisionClient)
    assert client.model == DEFAULT_QWENVL_PLUS_MODEL
    assert client.base_url == DEFAULT_QWENVL_BASE_URL
    assert client.thinking_budget == 500


def test_create_grounding_client_rejects_unknown_backend() -> None:
    from tools.yoloe26m.reference_pipeline import create_grounding_client

    with pytest.raises(ValueError, match="unsupported grounding backend"):
        create_grounding_client("other")


def test_manual_cli_parses_repeated_original_pixel_boxes() -> None:
    from tools.yoloe26m.visual_prompt import build_parser

    args = build_parser().parse_args(
        [
            "second.jpg",
            "--refer-image",
            "first.jpg",
            "--target-name",
            "blue basket",
            "--bbox",
            "10",
            "20",
            "110",
            "220",
            "--bbox",
            "300",
            "40",
            "500",
            "360",
        ]
    )

    assert args.source_image == Path("second.jpg")
    assert args.refer_image == Path("first.jpg")
    assert args.target_name == "blue basket"
    assert args.bbox == [[10, 20, 110, 220], [300, 40, 500, 360]]
    assert args.model == DEFAULT_MODEL
    assert args.imgsz == 640
    assert args.conf == 0.25
    assert args.iou == 0.7


def test_manual_cli_requires_at_least_one_bbox() -> None:
    from tools.yoloe26m.visual_prompt import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "second.jpg",
                "--refer-image",
                "first.jpg",
                "--target-name",
                "blue basket",
            ]
        )


def test_auto_cli_requires_explicit_target_and_selects_backend() -> None:
    from tools.yoloe26m.auto_refer_detect import build_parser

    args = build_parser().parse_args(
        [
            "second.jpg",
            "--refer-image",
            "first.jpg",
            "--target",
            "blue basket",
            "--backend",
            "qwenvl",
        ]
    )

    assert args.source_image == Path("second.jpg")
    assert args.refer_image == Path("first.jpg")
    assert args.target == "blue basket"
    assert args.backend == "qwenvl"
    assert args.codex_model == "gpt-5.6-sol"
    assert args.codex_reasoning_effort == "xhigh"
    assert args.qwen_thinking_budget == 500


def test_auto_cli_rejects_missing_target() -> None:
    from tools.yoloe26m.auto_refer_detect import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "second.jpg",
                "--refer-image",
                "first.jpg",
                "--backend",
                "codex",
            ]
        )


def test_auto_cli_rejects_unknown_backend() -> None:
    from tools.yoloe26m.auto_refer_detect import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "second.jpg",
                "--refer-image",
                "first.jpg",
                "--target",
                "blue basket",
                "--backend",
                "other",
            ]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("limitations"),
        lambda value: value.update({"unexpected": "unsafe"}),
    ],
)
def test_validate_grounding_enforces_complete_strict_object_schema(mutation) -> None:
    response = _ready_response([100, 100, 200, 200])
    mutation(response)

    with pytest.raises(GroundingValidationError, match="fields"):
        validate_grounding_response(response)


def test_manual_cli_no_longer_accepts_name_or_exist_ok() -> None:
    from tools.yoloe26m.visual_prompt import build_parser

    options = build_parser()._option_string_actions

    assert "--name" not in options
    assert "--exist-ok" not in options


def test_auto_cli_no_longer_accepts_name_or_exist_ok() -> None:
    from tools.yoloe26m.auto_refer_detect import build_parser

    options = build_parser()._option_string_actions

    assert "--name" not in options
    assert "--exist-ok" not in options
