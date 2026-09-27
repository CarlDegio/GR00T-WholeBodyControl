from pathlib import Path
import json
import re
import subprocess
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'render_manifest.json').read_text())
output = Path(manifest['output'])
source = ROOT.parent / 'exports/Full_demo_combined_v9_redacted_music.mp4'

def run(args):
    return subprocess.run(args, capture_output=True, check=True)

probe = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)]).stdout)
v, a = probe['streams']
assert (v['codec_type'], v['width'], v['height'], v['avg_frame_rate'], int(v['nb_frames'])) == ('video', 1920, 1080, '60/1', 10774)
assert (a['codec_type'], a['codec_name'], a['sample_rate'], a['channels']) == ('audio', 'aac', '48000', 2)
assert abs(float(probe['format']['duration']) - manifest['duration_seconds']) < .03
hashes = [run(['ffmpeg', '-v', 'error', '-i', str(p), '-map', '0:v:0', '-c:v', 'copy', '-f', 'hash', '-hash', 'sha256', '-']).stdout.decode().strip() for p in [source, output]]
assert hashes[0] == hashes[1], hashes

decoded = np.frombuffer(run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-map', '0:a:0', '-f', 'f32le', '-ar', '48000', '-ac', '2', '-']).stdout, dtype='<f4').reshape(-1, 2)
with wave.open(str(ROOT / 'looped_music.wav'), 'rb') as w:
    expected = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1, 2).astype(np.float32) / 32768
assert abs(len(decoded) - len(expected)) < 1024
loop = round(manifest['loop_restart_seconds'] * 48000)
checks = {}
for name, start, end in [('opening', 3*48000, 12*48000), ('loop_boundary', loop-48000, loop+48000), ('second_playthrough', loop+3*48000, loop+12*48000)]:
    correlation = float(np.corrcoef(expected[start:end].ravel(), decoded[start:end].ravel())[0, 1])
    assert correlation > .98, (name, correlation)
    checks[name] = correlation
stats_run = run(['ffmpeg', '-hide_banner', '-nostdin', '-xerror', '-i', str(output), '-vn', '-af', 'loudnorm=I=-24:LRA=7:TP=-3:print_format=json', '-f', 'null', '-'])
stats = json.loads(re.findall(r'\{[^{}]+\}', stats_run.stderr.decode())[-1])
assert -28 < float(stats['input_i']) < -21
assert float(stats['input_tp']) < -.5
# Measure half-second RMS windows spanning the former silent transition.
levels = {}
for name, path in [('previous', ROOT.parent / 'exports/Full_demo_combined_v10_creative_minds.mp4'), ('fixed', output)]:
    clip = np.frombuffer(run(['ffmpeg', '-v', 'error', '-ss', '138', '-i', str(path), '-t', '13', '-vn', '-f', 'f32le', '-ar', '48000', '-ac', '2', '-']).stdout, dtype='<f4').reshape(-1, 2)
    windows = clip[:len(clip)//24000*24000].reshape(-1,24000,2)
    db = 20*np.log10(np.sqrt(np.mean(windows**2, axis=(1,2)))+1e-12)
    levels[name] = dict(min_half_second_rms_db=float(db.min()), max_half_second_rms_db=float(db.max()))
assert levels['fixed']['min_half_second_rms_db'] > -36
assert levels['fixed']['min_half_second_rms_db'] > levels['previous']['min_half_second_rms_db'] + 12
result = dict(output=str(output), frames=10774, duration=float(probe['format']['duration']),
    video_unchanged_from_v9=True, video_packet_sha256=hashes[0], audio_full_decode='passed',
    loop_restart_seconds=manifest['loop_restart_seconds'], loop_audio_correlations=checks,
    loudness_lufs=float(stats['input_i']), true_peak_db=float(stats['input_tp']),
    transition_levels=levels, crossfade_seconds=manifest['crossfade_seconds'], bytes=output.stat().st_size)
(ROOT / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
