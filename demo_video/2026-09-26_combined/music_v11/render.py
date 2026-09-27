from pathlib import Path
import hashlib
import json
import re
import subprocess
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
SOURCE = BASE / 'exports/Full_demo_combined_v9_redacted_music.mp4'
PREVIOUS = BASE / 'music_v10'
TRACK = PREVIOUS / 'assets/bensound-creativeminds.mp3'
OUTPUT = BASE / 'exports/Full_demo_combined_v11_creative_minds_smooth_loop.mp4'
RATE = 48000
FRAMES = 10774
DURATION = FRAMES / 60

def run(args):
    return subprocess.run(args, check=True, capture_output=True).stdout

pcm = np.frombuffer(run(['ffmpeg', '-v', 'error', '-i', str(TRACK),
    '-f', 'f32le', '-ar', str(RATE), '-ac', '2', '-']), dtype='<f4').reshape(-1, 2)
source_stats = json.loads(re.findall(r'\{[^{}]+\}', (PREVIOUS / 'source_loudness.log').read_text())[-1])
gain_db = -24 - float(source_stats['input_i'])
gain = 10 ** (gain_db / 20)
total_samples = round(DURATION * RATE)
# Overlap the next playthrough before the original track fades to silence.
# 139.38 s selected from attack-envelope alignment near the ending.
loop_start = round(139.38 * RATE)
overlap = 4 * RATE
music = np.zeros((total_samples, 2), dtype=np.float32)
music[:loop_start] = pcm[:loop_start]
theta = np.linspace(0, np.pi / 2, overlap, dtype=np.float32)[:, None]
music[loop_start:loop_start+overlap] = (pcm[loop_start:loop_start+overlap] * np.cos(theta)
    + pcm[:overlap] * np.sin(theta))
music[loop_start+overlap:] = pcm[overlap:total_samples-loop_start]
music *= gain
music[:2 * RATE] *= np.linspace(0, 1, 2 * RATE, dtype=np.float32)[:, None]
music[-4 * RATE:] *= np.linspace(1, 0, 4 * RATE, dtype=np.float32)[:, None]
assert np.max(np.abs(music)) < 1
with wave.open(str(ROOT / 'looped_music.wav'), 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(RATE)
    w.writeframes(np.rint(music * 32767).astype('<i2').tobytes())

attribution = json.loads((PREVIOUS / 'assets/official_attribution.json').read_text())
assert attribution['error'] is False
credit = ('Music by Bensound.com\n'
          f"License code: {attribution['orfium_code']}\n"
          'Artist: Benjamin Tissot\n')
OUTPUT.with_suffix('.music.txt').write_text(
    'Creative Minds — Benjamin Tissot / Bensound\n\n'
    'Attribution for this video:\n' + credit + '\n'
    'Track: https://www.bensound.com/royalty-free-music/track/creative-minds\n'
    'FAQ: https://www.bensound.com/faq\n'
    'Use the attribution text in the published video description when using the free license.\n'
    'This code was obtained for this video; do not reuse it for unrelated videos.\n'
    'Playback: next cycle overlaps at 139.38 seconds with a 4-second equal-power crossfade.\n'
    'Overall 2-second fade-in and 4-second fade-out.\n')

run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(SOURCE),
     '-i', str(ROOT / 'looped_music.wav'), '-map', '0:v:0', '-map', '1:a:0',
     '-map_metadata', '-1', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
     '-ar', str(RATE), '-t', str(DURATION), '-movflags', '+faststart',
     '-metadata', 'title=Whole Body Control Demo',
     '-metadata', 'comment=Creative Minds — Benjamin Tissot / Bensound. ' + credit.replace('\n', ' '),
     str(OUTPUT)])
manifest = dict(output=str(OUTPUT), track=str(TRACK), duration_seconds=DURATION,
    loop_restart_seconds=loop_start/RATE, crossfade_seconds=overlap/RATE, track_samples=len(pcm), sample_rate=RATE,
    gain_db=gain_db, music_sha256=hashlib.sha256(TRACK.read_bytes()).hexdigest(),
    attribution=credit, video_encoding='stream copy from v9')
(ROOT / 'render_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
