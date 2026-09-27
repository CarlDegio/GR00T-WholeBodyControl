"""User-confirmed nearfield selection and clean-image audit; never score overlays."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.experiments.recording import read_events
from gear_sonic.experiments.results import trial_rows
from prepare import sha, write_json
from metrics import neighbor_distances

OUT = ROOT / 'experiment data/T2_垃圾袋入箱并搬箱/review/nearfield_alignment_20260921'
SOURCES = ROOT / 'outputs/nearfield_experiments/T2/align_vla'
BATCH = '20260922_T2_nearfield_user_selected_v2_success_v3_all'
OLD = 'experiment data/T2_垃圾袋入箱并搬箱/review/visual_handoff_20260918_head_chest'
CHECKPOINT = '/home/user/Project/openpi_sonic/checkpoints/pi05_g1_sonic_full_carry_box_big/pi05_g1_sonic_full_carry_box_big_179/29999'
V2 = '67c447cc2769'
V3 = {'e07a50bec078', 'b8b8042b7ba8'}


def read(p):
    return json.loads(Path(p).read_text())


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def select():
    records = []
    sources = []
    for p in sorted(SOURCES.glob('*/events.jsonl')):
        events = read_events(p)
        starts = {e['trial_id']: e for e in events if e['type'] == 'trial_start'}
        starts = {k: s for k, s in starts.items() if 1789920000000000000 <= s['wall_time_ns'] < 1790006400000000000}
        if not starts:
            continue
        cfg = yaml.safe_load(p.with_name('runtime.yaml').read_text())
        b = cfg['components']['base_pose']
        sid = cfg['components']['experiment']['session_id']
        version = 'V2' if sid == V2 else 'V3' if sid in V3 else 'other'
        if version == 'V2':
            assert (b['raw_head_target_distance_m'], b['raw_chest_target_distance_m']) == (1.2, 1.1)
        if version == 'V3':
            assert (b['raw_head_target_distance_m'], b['raw_chest_target_distance_m']) == (1.1, 1.15)
        sources.append(p)
        for row in trial_rows(events):
            if row['trial_id'] not in starts:
                continue
            start = starts[row['trial_id']]
            keep = version == 'V3' or (version == 'V2' and row['physical_success'] is True)
            records.append(dict(trial_id=row['trial_id'], session_id=sid, source_log=str(p),
                                version=version, head_target_m=b['raw_head_target_distance_m'],
                                chest_target_m=b['raw_chest_target_distance_m'],
                                selected=keep, physical_success=row['physical_success'], success=row['success'],
                                progress_numerator=row['progress_numerator'], progress_denominator=row['progress_denominator'],
                                generation=start['generation'], start_wall_ns=start['wall_time_ns'],
                                previous_exclude_reason=row['exclude_reason']))
    records.sort(key=lambda r:r['start_wall_ns'])
    assert len(records) == 47 and sum(r['selected'] for r in records) == 26
    manifest = dict(batch_id=BATCH, user_rule='第二版=②(1.20/1.10),只保留成功; 第三版=⑤(1.10/1.15),全部保留; 其余smoke',
                    physical_labels='Preserve latest independent operator annotations, pending is not failure',
                    version_selection_bias='V2 failures excluded by requested selection; combined success rate is not an unbiased evaluation.', trials=records)
    dest = OUT/'selection.json'
    if dest.exists():
        assert [(r['trial_id'],r['selected']) for r in read(dest)['trials']] == [(r['trial_id'],r['selected']) for r in records]
    else:
        write_json(dest, manifest)
    for p in sources:
        backup = OUT/'source_backups'/p.parent.name/'events.before_selection.jsonl'
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, backup)
        events = read_events(p)
        done = {e.get('trial_id') for e in events if e.get('batch_id') == BATCH}
        with p.open('a') as f:
            for r in records:
                if r['source_log'] != str(p) or r['trial_id'] in done:
                    continue
                values = dict(exclude_reason=None if r['selected'] else 'smoke_test',
                              nearfield_selection={'batch_id': BATCH, 'version': r['version'], 'selected': r['selected'],
                                                  'basis': manifest['user_rule']})
                event = dict(type='annotation', event_id=uuid.uuid4().hex, wall_time_ns=time.time_ns(),
                             monotonic_ns=time.monotonic_ns(), source='user_confirmed_offline_selection',
                             batch_id=BATCH, trial_id=r['trial_id'], generation=r['generation'], values=values)
                f.write(json.dumps(event, ensure_ascii=False)+'\n')
    # Register originals, never archive copies, with the existing archiver.
    policy = ROOT/'outputs/experiments/formal_results/selection_policy.json'
    backup = OUT/'source_backups/selection_policy.before.json'
    if not backup.exists():
        shutil.copy2(policy, backup)
    cfg = read(policy)
    cfg['additional_source_logs'] = sorted(set(cfg.get('additional_source_logs', [])) | {str(p) for p in sources})
    cfg.setdefault('trial_selections', {})[BATCH] = dict(review_file=str(dest),
        included_trial_ids=[r['trial_id'] for r in records if r['selected']],
        smoke_trial_ids=[r['trial_id'] for r in records if not r['selected']], user_instruction=manifest['user_rule'])
    write_json(policy, cfg)
    for p in sources:
        rows = {r['trial_id']: r for r in trial_rows(read_events(p))}
        for r in [r for r in records if r['source_log'] == str(p)]:
            now = rows[r['trial_id']]
            assert now['physical_success'] is r['physical_success']
            assert now['success'] is r['success']
            assert now['excluded'] is (not r['selected'])
            assert now['progress_numerator'] == r['progress_numerator']
            assert now['progress_denominator'] == r['progress_denominator']
    selected = [r for r in records if r['selected']]
    stats = dict(task='T2', method='near_dual', scope='2026-09-21 nearfield (batch equals near_dual cumulative)',
                 original_trials=len(records), kept=len(selected), smoke=len(records)-len(selected),
                 successes=sum(r['physical_success'] is True for r in selected),
                 failures=sum(r['physical_success'] is False for r in selected),
                 pending=sum(r['physical_success'] is None for r in selected),
                 source_hashes={str(p): sha(p) for p in sources},
                 validation='Outcomes and raw Progress numerator/denominator unchanged; 47 unique trial IDs')
    stats['labeled'] = stats['successes']+stats['failures']
    stats['success_rate'] = stats['successes']/stats['labeled']
    write_json(OUT/'selection_statistics.json',stats)
    print(json.dumps({k:v for k,v in stats.items() if k!='source_hashes'},ensure_ascii=False))


def git_bytes(path):
    blob = subprocess.check_output(['git','show', f'agent_full:{path}'], cwd=ROOT)
    if blob.startswith(b'version https://git-lfs.github.com/spec/v1'):
        digest = blob.decode().split('oid sha256:')[1].splitlines()[0]
        obj = ROOT/'.git/lfs/objects'/digest[:2]/digest[2:4]/digest
        data = obj.read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        return data
    return blob


def references():
    dest = OUT/'references'
    dest.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(git_bytes(OLD+'/analysis_config.json'))
    prov = json.loads(git_bytes(OLD+'/features/provenance.json'))
    samples = json.loads(git_bytes(OLD+'/inputs/samples.json'))
    data = git_bytes(OLD+'/features/embeddings.npz')
    assert hashlib.sha256(data).hexdigest() == prov['feature_file_sha256']
    f = np.load(io.BytesIO(data))
    assert list(f['sample_ids']) == [s['sample_id'] for s in samples]
    indices = [i for i,s in enumerate(samples) if s['split']=='train' and s['view']=='head']
    assert len(indices) == 179 and [samples[i]['episode_id'] for i in indices] == list(range(179))
    np.savez_compressed(dest/'head_reference_features.npz', sample_ids=f['sample_ids'][indices],
                        spatial=f['spatial'][indices], global_mean=f['global_mean'][indices])
    refs = []
    for i in indices:
        s = dict(samples[i])
        name = Path(s['image']).name
        p = dest/'training_frames'/name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(git_bytes(OLD+'/inputs/training_frames/'+name))
        assert sha(p)==s['image_sha256']
        s['original_image'] = s['image']; s['image']=str(p)
        refs.append(s)
    write_json(dest/'samples.json', refs)
    write_json(dest/'original_provenance.json', prov)
    write_json(dest/'recovery_provenance.json', dict(source_branch='agent_full',
               source_commit=subprocess.check_output(['git','rev-parse','agent_full'],cwd=ROOT,text=True).strip(),
               source_path=OLD, original_feature_sha256=prov['feature_file_sha256'],
               head_subset_sha256=sha(dest/'head_reference_features.npz'), n=179,
               training_images_verified=179))
    cfg.update(checkpoint=CHECKPOINT, camera='ego_view',analysis='Nearfield T2 selected cohort; clean post-alignment images',
               pre_image_limitation='Saved initial.jpg includes segmentation/edge overlays; not eligible as clean primary feature input.',
               selection_manifest=str(OUT/'selection.json'), feature_file=str(OUT/'features/embeddings.npz'))
    write_json(OUT/'analysis_config.json',cfg)
    print('Recovered and SHA256-verified 179 training head images/features from agent_full.')


def audit():
    records = [r for r in read(OUT/'selection.json')['trials'] if r['selected']]
    samples=[]; audited=[]
    for r0 in records:
        r=dict(r0); source=Path(r['source_log']); es=[e for e in read_events(source) if e.get('trial_id')==r['trial_id']]
        start=next(e for e in es if e['type']=='trial_start')
        first_vla=next((e['wall_time_ns'] for e in es if e['type']=='vla_command' and e.get('command')=='start_vla_task'),float('inf'))
        gates=[e for e in es if e['type']=='gate' and e.get('gate')=='align' and e.get('applied')=='allow' and e['wall_time_ns']<first_vla]
        r['pre_diagnostic']=None; r['post_image']=None; r['clean_pair_available']=False
        initials=sorted(source.parent.glob(f'base_pose/*_g{r["generation"]*1000000+1}/diagnostic_images/ego_view/initial.jpg'))
        if initials:
            assert len(initials)==1
            r['pre_diagnostic']=str(initials[0]); r['pre_diagnostic_sha256']=sha(initials[0])
            r['pre_eligibility']='Diagnostic writer applies target mask and edge lines; excluded from clean-primary inputs'
        if gates:
            g=gates[-1]
            checks=[e for e in es if e['type']=='check' and e.get('event_id') in g['check_ids'] and e.get('camera')=='head']
            assert len(checks)==1
            check=checks[0]; stamp=check['observation']['source_timestamp_ns']
            controllers=[e for e in es if e.get('channel')=='base_pose_status' and e.get('skill_id')==check['skill_id'] and e.get('reason')=='aligned']
            assert controllers and start['wall_time_ns']<controllers[-1]['wall_time_ns']<=stamp<first_vla
            p=Path(check['image']); assert p.exists() and Image.open(p).size==(640,480)
            copied=OUT/'inputs/post_head'/f'{r["trial_id"]}.png';copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,copied);assert sha(copied)==sha(p)
            r.update(post_image=str(copied),post_original_image=str(p),post_check_event=check,
                     post_seconds_before_vla=(first_vla-stamp)/1e9, post_seconds_after_aligned=(stamp-controllers[-1]['wall_time_ns'])/1e9)
            samples.append(dict(sample_id='post_'+r['trial_id'],trial_id=r['trial_id'],split='experiment',stage='post',view='head',
                                image=str(copied),image_sha256=sha(copied)))
        else:
            r['missing_post_reason']='No completed ALIGN head check before VLA; no post-image substitution'
        audited.append(r)
    write_json(OUT/'inputs/trials.json',audited);write_json(OUT/'inputs/samples.json',samples)
    write_csv(OUT/'image_availability.csv',[{k:r.get(k) for k in ['trial_id','version','physical_success','pre_diagnostic','post_image','clean_pair_available','missing_post_reason']} for r in audited])
    # Review contact sheets are diagnostic evidence, not encoder inputs.
    for page,offset in enumerate(range(0,len(audited),8),1):
        sheet=Image.new('RGB',(1024,8*218),(247,249,250)); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(audited[offset:offset+8]):
            draw.text((8,j*218+3),f'{r["version"]} {r["trial_id"]}  physical_success={r["physical_success"]}', fill=(20,35,50))
            for col,key in enumerate(['pre_diagnostic','post_image']):
                if r.get(key):
                    im=Image.open(r[key]).convert('RGB');im.thumbnail((256,192));sheet.paste(im,(col*512+120,j*218+23))
                draw.text((col*512+4,j*218+30),'DIAGNOSTIC PRE' if col==0 else 'CLEAN POST',fill=(120,50,50))
        sheet.save(OUT/f'image_audit_{page:02d}.jpg',quality=92)
    print('selected',len(audited),'clean post',len(samples),'saved diagnostic pre',sum(bool(r['pre_diagnostic']) for r in audited))


def analyze_post():
    f=np.load(OUT/'features/embeddings.npz');ref=np.load(OUT/'references/head_reference_features.npz')
    prov=read(OUT/'features/provenance.json');old=read(OUT/'references/original_provenance.json')
    assert prov['vision_bfloat16_parameter_sha256']==old['vision_bfloat16_parameter_sha256']
    samples=read(OUT/'inputs/samples.json');assert list(f['sample_ids'])==[s['sample_id'] for s in samples]
    assert sha(OUT/'features/embeddings.npz')==prov['feature_file_sha256']
    trials={r['trial_id']:r for r in read(OUT/'inputs/trials.json')}; rows=[]
    ds,order=neighbor_distances(f['spatial'],ref['spatial']);dg,_=neighbor_distances(f['global_mean'],ref['global_mean'])
    for i,s in enumerate(samples):
        assert sha(s['image'])==s['image_sha256']
        t=trials[s['trial_id']]
        rows.append(dict(trial_id=s['trial_id'],version=t['version'],physical_success=t['physical_success'],post_head_spatial=float(ds[i]),
                         post_head_global=float(dg[i]),nearest_3_training_episodes=','.join(str(int(j)) for j in order[i]),
                         pre_head_spatial=None,delta=None,paired_status='missing_clean_pre',image=s['image']))
    write_csv(OUT/'results/post_alignment_distances.csv',rows)
    write_json(OUT/'results/post_alignment_distances.json',rows)
    write_json(OUT/'results/status.json',dict(selected_trials=26,scored_clean_post=len(rows),
        clean_pairs=0,limitation='No unannotated pre-alignment head frames found. No feature change or trend estimated.',
        references=179,vision_parameter_hash=prov['vision_bfloat16_parameter_sha256']))
    print('Scored',len(rows),'clean post images. Primary paired analysis awaits clean pre images.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','references','audit','analyze_post']);args=p.parse_args()
    globals()[args.action]()
