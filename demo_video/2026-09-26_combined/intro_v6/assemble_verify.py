"""Replace v5's first 300 frames with the supplied five-second intro losslessly."""
from pathlib import Path
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
EXPORTS = BASE.parent / 'exports'
SOURCE = EXPORTS / 'Full_demo_combined_v5_redacted.mp4'
INTRO = EXPORTS / 'Intro_5s_B_champagne_bold.mp4'
OUTPUT = EXPORTS / 'Full_demo_combined_v6_redacted.mp4'


def run(args):
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', *args], check=True)


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def frame_hashes(path):
    return [line.rsplit(',', 1)[-1].strip() for line in path.read_text().splitlines()
            if line and not line.startswith('#')]


old_receipt = json.loads((BASE.parent / 'layout_v5/verification.json').read_text())
assert sha(SOURCE) == old_receipt['sha256']
intro_stream = probe(INTRO)['streams'][0]
assert (intro_stream['width'], intro_stream['height'], intro_stream['r_frame_rate'],
        int(intro_stream['nb_frames'])) == (1920, 1080, '60/1', 300)

tail = BASE / 'v5_after_intro.mp4'
run(['-ss', '5', '-i', str(SOURCE), '-map', '0:v:0', '-an', '-c', 'copy',
     '-map_metadata', '-1', str(tail)])
assert int(probe(tail)['streams'][0]['nb_frames']) == 10474
concat = BASE / 'concat.txt'
concat.write_text(f"file '{INTRO}'\nduration 5.000000000000\nfile '{tail}'\n")
run(['-f', 'concat', '-safe', '0', '-i', str(concat), '-map', '0:v:0', '-c', 'copy',
     '-an', '-map_metadata', '-1', '-movflags', '+faststart', str(OUTPUT)])
print('v6 assembled without re-encoding.', flush=True)

info = probe(OUTPUT)
stream = info['streams'][0]
assert len(info['streams']) == 1
assert (stream['width'], stream['height'], stream['r_frame_rate'], int(stream['nb_frames'])) == (1920, 1080, '60/1', 10774)
assert abs(float(info['format']['duration']) - 10774 / 60) < .002
for label, path in [('intro', INTRO), ('output', OUTPUT)]:
    with (BASE / f'{label}_decode.log').open('w') as log:
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-n', '-threads', '4',
                        '-i', str(path), '-map', '0:v:0', '-f', 'framemd5',
                        str(BASE / f'{label}.framemd5')], stderr=log, check=True)
    print(f'{label}: full decode passed.', flush=True)
intro_hashes = frame_hashes(BASE / 'intro.framemd5')
old_hashes = frame_hashes(BASE.parent / 'layout_v5/output.framemd5')
new_hashes = frame_hashes(BASE / 'output.framemd5')
assert len(intro_hashes) == 300
assert len(new_hashes) == 10774
assert new_hashes[:300] == intro_hashes
assert new_hashes[300:] == old_hashes[300:]
receipt = dict(source=str(SOURCE), intro=str(INTRO), output=str(OUTPUT),
               resolution=[1920, 1080], fps=60, frames=10774, duration_seconds=10774 / 60,
               intro_identical_frames=300, remainder_identical_frames=10474,
               full_decode='passed', bytes=OUTPUT.stat().st_size, sha256=sha(OUTPUT))
(BASE / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
