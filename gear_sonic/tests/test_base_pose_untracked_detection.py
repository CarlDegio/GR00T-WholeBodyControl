"""Detection-only BasePose must accept segmentation results without track IDs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from gear_sonic.utils.inference.base_pose.servo import YoloePersistentTracker
from gear_sonic.utils.inference.base_pose.dual_servo import _select_tracked_instances


def detector(tracking_enabled, second_target_score=.09):
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    boxes = torch.tensor([
        [2, 2, 14, 14, .165, 0],
        [16, 2, 30, 14, second_target_score, 0],
        [2, 16, 14, 30, .24, 1],
        [16, 16, 30, 30, .4, 1],
    ])
    masks = torch.zeros((4, 32, 32))
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        masks[i, y1:y2, x1:x2] = 1
    result = Results(rgb, path='recorded.jpg', names={0: 'cardboard box', 1: 'desk'},
                     boxes=boxes, masks=masks)
    calls = []

    def predict(image, **kwargs):
        calls.append(('predict', kwargs))
        return [result]

    def track(image, **kwargs):
        calls.append(('track', kwargs))
        return [result]  # Detection exists but BoT-SORT has not assigned an ID.

    adapter = object.__new__(YoloePersistentTracker)
    adapter.model = SimpleNamespace(predict=predict, track=track)
    adapter.class_names = ('cardboard box', 'desk')
    adapter.confidence = .25
    adapter.target_confidence = .1
    adapter.imgsz = 640
    adapter.device = 'cpu'
    adapter.tracking_enabled = tracking_enabled
    return adapter, rgb, calls


def test_detection_only_accepts_mask_without_id_and_preserves_role_thresholds():
    adapter, rgb, calls = detector(False)
    instances = adapter._track_on_current_cuda_stream(rgb)
    assert len(instances) == 2
    assert [x.class_index for x in instances] == [0, 1]
    assert [x.confidence for x in instances] == pytest.approx([.165, .4])
    assert all(x.track_id is None and x.mask.any() for x in instances)
    target, yaw, _, _ = _select_tracked_instances(
        instances, target_id=None, yaw_align_target_id=None,
        reuse_target_as_yaw_align_target=False,
    )
    assert target.bbox_xyxy == (2, 2, 14, 14)
    assert yaw.bbox_xyxy == (16, 16, 30, 30)
    assert calls[0][0] == 'predict'
    assert 'tracker' not in calls[0][1]


def test_tracking_mode_still_rejects_unconfirmed_detections():
    adapter, rgb, calls = detector(True)
    assert adapter._track_on_current_cuda_stream(rgb) == []
    assert calls[0][0] == 'track'


def test_detection_only_selects_best_target_and_reuses_it_for_yaw():
    adapter, rgb, _ = detector(False, second_target_score=.6)
    instances = adapter._track_on_current_cuda_stream(rgb)
    target, yaw, _, _ = _select_tracked_instances(
        instances, target_id=None, yaw_align_target_id=None,
        reuse_target_as_yaw_align_target=True,
    )
    assert target is not None
    assert target.bbox_xyxy == (16, 2, 30, 14)
    assert yaw is target
    assert target.track_id is None


@pytest.mark.parametrize('mission', [
    't1_bottle_to_basket.yaml', 't2_bag_and_box.yaml',
    't3_bowl_to_bin.yaml', 'bag_then_bin.yaml',
])
def test_all_missions_load_detection_only_for_base_pose(tmp_path, mission):
    import yaml
    from gear_sonic.experiments.config import CONFIGS, resolve
    from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config

    runtime, missing = resolve(CONFIGS / 'full_vln.yaml', mission_path=CONFIGS.parent / 'missions' / mission)
    assert not missing
    path = tmp_path / 'runtime.yaml'
    path.write_text(yaml.safe_dump(runtime))
    assert load_base_pose_config(str(path)).raw_yoloe_tracking is False
    # Older frozen profiles do not declare the new field either.
    runtime['components']['base_pose'].pop('raw_yoloe_tracking')
    path.write_text(yaml.safe_dump(runtime))
    assert load_base_pose_config(str(path)).raw_yoloe_tracking is False
