"""Assemble the corrected T4 source, reordered demos, and two-second end card."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'exports'
TASK = ROOT / 'outputs/demo_edits/T2_T4_20260926_combined_v7'

subprocess.run([
    'ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-n',
    '-f', 'concat', '-safe', '0', '-i', str(TASK / 'concat.txt'),
    '-map', '0:v:0', '-map_metadata', '-1', '-map_chapters', '-1',
    '-c', 'copy', '-movflags', '+faststart', str(TASK / 'Task_demos_v7.mp4'),
], check=True)

parts = [
    ('Cover', ROOT / 'demo_video/2026-09-26_cover/exports/cover_16_5s.mp4', 300),
    ('Comparison preview', ROOT / 'demo_video/2026-09-25/exports/00_combined_preview_v13.mp4', 2418),
    ('OVMM', ROOT / 'outputs/demos/ovmm_place_20260926/GATE_vs_Official_RL_OVMM_4x1.mp4', 600),
    ('Task demos', TASK / 'Task_demos_v7.mp4', 7336),
    ('Thank you for watching!', OUT / 'thank_you_2s.mp4', 120),
]
cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-n']
for _, path, _ in parts:
    cmd += ['-threads', '2', '-i', str(path)]
filters = [
    f'[{i}:v:0]setpts=PTS-STARTPTS,scale=1920:1080:flags=lanczos,setsar=1,fps=60[v{i}]'
    for i in range(len(parts))
]
filters.append(''.join(f'[v{i}]' for i in range(len(parts))) + 'concat=n=5:v=1:a=0[v]')
cmd += [
    '-filter_complex_threads', '4', '-filter_complex', ';'.join(filters),
    '-map', '[v]', '-map_metadata', '-1', '-map_chapters', '-1',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '8',
    '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
    str(OUT / 'Full_demo_combined_v2.mp4'),
]
with (OUT / 'encode_v2.log').open('w') as log:
    subprocess.run(cmd, stderr=log, check=True)
cursor = 0
segments = []
for name, path, frames in parts:
    segments.append(dict(name=name, source=str(path), start_seconds=cursor/60,
                         duration_seconds=frames/60, frames=frames))
    cursor += frames
(OUT / 'sources_v2.json').write_text(json.dumps(dict(
    fps=60, resolution=[1920, 1080], total_frames=cursor,
    duration_seconds=cursor/60, segments=segments,
    replacement_source='/home/user/.codex/attachments/231f45a1-a5be-47fe-8e6e-43457ac3a18e/0599a2e8b259491efe3e62a74a1b6f8e.mp4',
    t4_source_offset_seconds=0,
), indent=2) + '\n')
print('Assembly complete:', cursor, 'frames', flush=True)
