"""Freeze the approved image cohort and latest source labels without mutating them."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('T1_VISUAL_OUT', ROOT / 'experiment data/T1_药瓶入篮/review/visual_handoff_20260918'))
HEAT = ROOT / 'experiment data/T1_药瓶入篮/review/heatmap_20260915_four_methods_5px'
SUMMARY = ROOT / 'experiment data/T1_药瓶入篮/summary'
CHECKPOINT = Path('/home/user/Project/openpi_sonic/checkpoints/pi05_g1_sonic_full_collect_pill_mix/pi05_g1_sonic_full_collect_pill_mix_289/29999')
METHODS = ['full_vln', 'nav_direct_vla', 'geometric_vla', 'baselinelavira']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    from metrics import outcome_fraction
    samples = json.loads((HEAT/'samples.json').read_text())
    pre = json.loads((HEAT/'preflight.json').read_text())
    totals = {r['trial_id']:r for r in csv.DictReader((SUMMARY/'task1_total.csv').open(encoding='utf-8-sig'))}
    archived = {r['trial_id']:r for r in json.loads((SUMMARY/'statistics.json').read_text())['trials']}
    geometry = list(csv.DictReader((HEAT/'results/all_training/trials.csv').open()))
    gs = {r['trial_id']:r for r in geometry}
    by_trial = {s['trial_id']:s for s in samples if s['split']=='experiment'}
    seen = set()
    for s in samples:
        assert s['sample_id'] not in seen
        seen.add(s['sample_id'])
        assert sha(s['image']) == s['image_sha256'], s['image']
        assert s['image_size']==[640,480]
    source_hashes = {}
    trials=[]
    for t in pre['trials']:
        tid=t['trial_id']; row=totals[tid]; log=Path(row['source_log'])
        source_hashes[str(log)]=sha(log)
        values={}; ann_events=[]; starts=[]
        for line in log.read_text().splitlines():
            if not line.strip(): continue
            e=json.loads(line)
            if e.get('trial_id')!=tid: continue
            if e.get('type')=='annotation':
                values.update(e['values'])
                ann_events.append({'event_id':e.get('event_id'), 'wall_time_ns':e.get('wall_time_ns'), 'values':e['values']})
            if e.get('type')=='trial_start': starts.append(e)
        success=values.get('success')
        assert success is (row['success']=='True'), (tid, success, row['success'])
        assert success is archived[tid]['physical_success']
        assert not values.get('exclude_reason'), (tid, values.get('exclude_reason'))
        num=values.get('progress'); den=values.get('progress_denominator')
        fraction=outcome_fraction(success,num,den)
        assert fraction is not None and abs(fraction-float(row['progress']))<1e-12
        sample=by_trial.get(tid)
        g=gs[tid]
        assert bool(sample)==(g['has_chest_snapshot']=='True')
        trials.append({
            'trial_id':tid,'method':row['method'],'case_id':row['case_id'],
            'sample_id':sample['sample_id'] if sample else None,
            'image':sample['image'] if sample else None,
            'success':success,'raw_progress_numerator':num,'raw_progress_denominator':den,
            'raw_progress_ratio':num/den if num is not None and den else None,
            'report_progress':fraction,
            'progress_basis':'success_full_completion' if success else 'explicit_user_fraction',
            'success_basis':values.get('success_basis','latest_independent_annotation'),
            'source_log':str(log),'annotation_events':ann_events,
            'intervention':archived[tid]['intervention'],
            'terminal_reason':archived[tid]['terminal_reason'],
            'has_image':bool(sample),'vla_started':g['vla_started']=='True',
            'pose_valid':g['pose_valid']=='True','strict_pose_valid':g['strict_pose_valid']=='True',
            'geometry_distance':float(g['normalized_distance']) if g['normalized_distance'] else None,
            'geometry_neighbor_episode':int(g['nearest_episode_id']) if g['nearest_episode_id'] else None,
            'geometry_position_distance_m':float(g['nearest_position_distance_m']) if g['nearest_position_distance_m'] else None,
            'geometry_heading_difference_deg':float(g['nearest_heading_difference_deg']) if g['nearest_heading_difference_deg'] else None,
            'inferred_corners':g['inferred_corners'],'off_frame_estimated':g['off_frame_estimated']=='True',
            'geometry_failure_reason':g['failure_reason'],
        })
    assert len(trials)==40 and sum(t['has_image'] for t in trials)==36
    assert sum(t['pose_valid'] for t in trials)==33
    for p in [HEAT/'samples.json',HEAT/'preflight.json',HEAT/'analysis_config.json',HEAT/'results/poses.json',
              HEAT/'results/calibration.json',HEAT/'results/all_training/trials.csv',
              HEAT/'results/from_00145/trials.csv',SUMMARY/'task1_total.csv',SUMMARY/'statistics.json',
              SUMMARY/'task1_metrics.json',ROOT/'outputs/experiments/formal_results/selection_policy.json']:
        source_hashes[str(p)]=sha(p)
    config={
        'created_at':datetime.now(timezone.utc).isoformat(),'task':'T1','camera':'chest_view',
        'checkpoint':str(CHECKPOINT),'checkpoint_confirmation':'user_confirmed_exact_path_in_this_conversation',
        'heatmap_source':str(HEAT),'methods':METHODS,
        'training_reference':'available cached episodes 0..237; no assertion that cache covers all checkpoint training episodes',
        'feature_layer':'SigLIP So400m/14 projected output PaliGemma.img; before language fusion',
        'preprocess':'OpenCV RGB decode; OpenPI resize_with_pad uint8 224x224; float32 /255*2-1; inference train=False',
        'descriptor_primary':'3x4 spatial cells; each averages 4x4 patches in valid 12x16 grid; per-cell L2 then concatenate/L2',
        'descriptor_sensitivity':'global mean of same 12x16 non-padding tokens, L2 normalize',
        'distance':'mean cosine distance of nearest 3 distinct training episodes; nearest 1 retained',
        'training_baseline':'leave-one-episode-out; same frozen encoder and preprocessing',
        'references':['available_238','common_geometry_197','from_00145_93'],
        'random_seed':20260918,'bootstrap_resamples':2000,
        'inference':'vision encoder only; no action generation or robot connections',
        'claims':'exploratory associations; no causal, success prediction, attention or full-distribution matching claims',
    }
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'analysis_config.json',config)
    write_json(OUT/'inputs/samples.json',samples)
    write_json(OUT/'inputs/trials.json',trials)
    write_json(OUT/'inputs/source_hashes.json',source_hashes)
    write_json(OUT/'inputs/formal_method_counts.json',json.loads((SUMMARY/'task1_metrics.json').read_text()))
    print(json.dumps({'out':str(OUT),'images':len(samples),'trials':len(trials),'independent_successes':sum(t['success'] for t in trials),'source_hashes':len(source_hashes)},ensure_ascii=False))


if __name__=='__main__': main()
