"""Freeze provenance, decode exact first frames, and prepare blinded annotation jobs."""
from pathlib import Path
from paths import ROOT
import argparse
import hashlib
import json
import random
import shutil

import av
from PIL import Image




def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-only', action='store_true')
    args = parser.parse_args()
    pre = json.loads((ROOT/'preflight.json').read_text())
    sources = [('train', i) for i in range(238)] + [('experiment', t['trial_id']) for t in pre['trials'] if t['first_policy_chest_frame'] and t['vla_first_action']]
    shuffled = list(sources)
    random.Random(9026).shuffle(shuffled)
    blind = {key: f'img_{i:04d}' for i,key in enumerate(shuffled)}
    (ROOT/'frames').mkdir(exist_ok=True)
    (ROOT/'jobs').mkdir(exist_ok=True)
    (ROOT/'annotations').mkdir(exist_ok=True)
    (ROOT/'provenance').mkdir(exist_ok=True)
    calib_path = Path(pre['calibration']['local_path'])
    frozen = ROOT/'provenance/camera_intrinsics.json'
    if not frozen.exists(): shutil.copyfile(calib_path, frozen)
    experiments=[]
    for t in pre['trials']:
        frame=t['first_policy_chest_frame']
        if frame is None or not t['vla_first_action']: continue
        source=Path(frame['rgb']); sid=blind[('experiment',t['trial_id'])]
        output=ROOT/'frames'/f'{sid}.png'
        with Image.open(source) as im:
            im.convert('RGB').save(output)
            shape=im.size
        experiments.append({'sample_id':sid,'split':'experiment','trial_id':t['trial_id'],'method':t['method'],'case_id':t['case_id'],'image':str(output),'image_size':list(shape),'source_image':str(source),'source_sha256':sha256(source),'image_sha256':sha256(output),'snapshot':frame,'outcome':t['annotation_values']})
    (ROOT/'provenance/experiments.json').write_text(json.dumps(experiments,ensure_ascii=False,indent=2)+'\n')
    job=[{'sample_id':s['sample_id'],'image':s['image'],'image_size':s['image_size']} for s in experiments]
    (ROOT/'jobs/experiments.json').write_text(json.dumps(job,indent=2)+'\n')
    if args.experiments_only:
        print(json.dumps({'experimental_frames':len(experiments),'job':str(ROOT/'jobs/experiments.json')}));return
    train=[]
    folder=ROOT/'data/CollectPillMix/videos/chunk-000/observation.images.chest_view'
    for i in range(238):
        video=folder/f'episode_{i:06d}.mp4';sid=blind[('train',i)];output=ROOT/'frames'/f'{sid}.png'
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            frame=next(container.decode(stream))
            image=frame.to_image().convert('RGB');image.save(output)
            pts=frame.pts;time_base=str(frame.time_base);timestamp=float(frame.time) if frame.time is not None else None
            width,height=image.size
        train.append({'sample_id':sid,'split':'train','episode_id':i,'image':str(output),'image_size':[width,height],'source_video':str(video),'source_sha256':sha256(video),'image_sha256':sha256(output),'frame_index':0,'frame_pts':pts,'frame_time_base':time_base,'frame_time_s':timestamp})
    (ROOT/'provenance/training.json').write_text(json.dumps(train,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'samples.json').write_text(json.dumps(train+experiments,ensure_ascii=False,indent=2)+'\n')
    for offset in range(0,len(train),40):
        batch=train[offset:offset+40]
        job=[{'sample_id':s['sample_id'],'image':s['image'],'image_size':s['image_size']} for s in batch]
        (ROOT/'jobs'/f'train_{offset:03d}_{offset+len(batch)-1:03d}.json').write_text(json.dumps(job,indent=2)+'\n')
    print(json.dumps({'training_frames':len(train),'experimental_frames':len(experiments),'subset_145':sum(s['episode_id']>=145 for s in train),'samples_manifest':str(ROOT/'samples.json')}))


if __name__=='__main__':main()
