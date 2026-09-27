"""Restore full-frame handheld views using the exact approved edit timeline."""
from pathlib import Path
import json
import math
import subprocess
import sys
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
RAW = {
    't2': '/home/user/.codex/attachments/4c902372-413f-47fc-8b11-3802d941f396/b80a27efc60e04b9dd7ec8cbdb30158a.mp4',
    't4': '/home/user/.codex/attachments/231f45a1-a5be-47fe-8e6e-43457ac3a18e/0599a2e8b259491efe3e62a74a1b6f8e.mp4',
}
TASKS = {'t2': ('T2_20260926_v8', 3318), 't4': ('T4_20260926_v7', 7080)}


def source_times(task):
    data = json.loads((ROOT / 'outputs/demo_edits' / TASKS[task][0] / 'timeline.json').read_text())
    result = []
    for seg in data['segments']:
        result.extend(seg['start'] + (seg['end'] - seg['start']) * j / seg['frames'] for j in range(seg['frames']))
    assert len(result) == data['total_frames']
    return result


class Reader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.index = -1
        self.image = None

    def get(self, time):
        target = max(0, round(time * self.fps))
        if target < self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.index = target - 1
        while self.index < target:
            assert self.cap.grab(), target
            self.index += 1
        ok, image = self.cap.retrieve()
        assert ok
        self.image = image
        return image


def main(task):
    cv2.setNumThreads(2)
    times = source_times(task)
    masks = json.loads((BASE.parent / 'privacy_redaction_v4/merged_masks.json').read_text())['frames']
    reader = Reader(RAW[task])
    detector = cv2.FaceDetectorYN_create(str(BASE / 'assets/face_detection_yunet_2023mar.onnx'), '', (960, 544), .65, .3, 5000)
    reviewed_path = BASE / f'{task}_reviewed_additional_masks.json'
    reviewed = json.loads(reviewed_path.read_text()) if reviewed_path.exists() else None
    cmd = ['ffmpeg', '-v', 'warning', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', '960x540', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
           '-crf', '14', '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(BASE / f'{task}_main.mp4')]
    extras, mapping = {}, []
    review = BASE / f'{task}_exposed_region_review'
    review.mkdir(exist_ok=True)
    with (BASE / f'{task}_main_encode.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        for i, t in enumerate(times):
            raw = reader.get(t)
            fid = TASKS[task][1] + i * 2
            boxes = []
            # Original composition fits 960x544 into 1271x720 at x=4,
            # then scales to 1920x1080. Invert that mapping to raw pixels.
            for n in (fid, fid + 1):
                for x1, y1, x2, y2 in masks.get(str(n), []):
                    if x1 >= 1428: continue  # Camera-inset masks stay on their existing feeds.
                    boxes.append([max(0, math.floor((x1 - 6) * 960 / 1906.5) - 2),
                                  max(0, math.floor(y1 * 544 / 1080) - 2),
                                  min(960, math.ceil((x2 - 6) * 960 / 1906.5) + 2),
                                  min(544, math.ceil(y2 * 544 / 1080) + 2)])
            faces = None
            if reviewed is None:
                _, faces = detector.detect(raw)
            extra = []
            if faces is not None:
                for f in faces:
                    x, y, w, h = map(float, f[:4])
                    # Only add masks for areas formerly hidden by the old title/sidebar.
                    if not (x + w > 705 or (y < 55 and x < 450)): continue
                    extra.append([max(0, int(x - .25*w)), max(0, int(y - .3*h)),
                                  min(960, math.ceil(x + 1.25*w)), min(544, math.ceil(y + 1.25*h))])
            if reviewed is not None:
                extra = reviewed.get(str(i), [])
            if extra:
                extras[str(i)] = extra
                if i % 3 == 0:
                    shot = raw.copy()
                    for x1, y1, x2, y2 in extra: cv2.rectangle(shot, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.imwrite(str(review / f'{i:04d}.jpg'), shot)
            boxes.extend(extra)
            out = raw.copy()
            if boxes:
                mosaic = cv2.resize(cv2.resize(raw, (48,27), interpolation=cv2.INTER_AREA), (960,544), interpolation=cv2.INTER_NEAREST)
                for x1,y1,x2,y2 in boxes: out[y1:y2,x1:x2] = mosaic[y1:y2,x1:x2]
            # Two raw pixels at top/bottom are the only framing crop (16:9 fill).
            proc.stdin.write(out[2:542].tobytes())
            mapping.append(dict(frame=i, original_frame=round(t * 30), original_seconds=t, prior_global_frame=fid))
            if i % 300 == 0: print(task, i, '/', len(times), flush=True)
        proc.stdin.close()
        assert proc.wait() == 0
    reader.cap.release()
    (BASE / f'{task}_additional_masks.json').write_text(json.dumps(extras))
    (BASE / f'{task}_source_mapping.json').write_text(json.dumps(mapping))
    print(task, 'full-frame background complete; additional masked frames:', len(extras), flush=True)


if __name__ == '__main__':
    for task in sys.argv[1:] or ['t2','t4']: main(task)
