"""Check source integrity, annotation provenance, accounting, and artifact links."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

from PIL import Image

from paths import ROOT


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    samples = json.loads((ROOT / 'samples.json').read_text())
    poses = json.loads((ROOT / 'results/poses.json').read_text())
    pre = json.loads((ROOT / 'preflight.json').read_text())
    config = json.loads((ROOT / 'analysis_config.json').read_text())
    extension = config.get('cohort_extension', {})
    expected_experiments = extension.get('expected_experiment_samples', 12)
    expected_trials = extension.get('expected_formal_trials', 13)
    expected_started = extension.get('expected_vla_started', 12)
    assert len(samples) == len(poses) == 238 + expected_experiments
    assert {s['sample_id'] for s in samples} == {p['sample_id'] for p in poses}
    train = [s for s in samples if s['split'] == 'train']
    assert {s['episode_id'] for s in train} == set(range(238))
    assert len([s for s in train if s['episode_id'] >= 145]) == 93
    assert len(pre['trials']) == expected_trials
    assert sum(t['vla_first_action'] for t in pre['trials']) == expected_started
    assert len({t['trial_id'] for t in pre['trials']}) == len(pre['trials'])
    assert not {t['trial_id'] for t in pre['trials']} & {t['trial_id'] for t in pre['excluded_trials']}
    first_input_proof = []
    for trial in pre['trials']:
        events = [json.loads(line) for line in Path(trial['events']).read_text().splitlines() if line.strip()]
        if trial['trial_id'] in extension.get('added_trial_ids', []):
            from gear_sonic.experiments.results import trial_rows
            actual = next(row for row in trial_rows(events) if row['trial_id'] == trial['trial_id'])
            assert not actual['excluded']
            assert actual['case_id'] == trial['case_id']
            assert actual['success'] == trial['annotation_values'].get('success')
        def matching(kind):
            return sorted([e for e in events if e['type']==kind and e.get('trial_id')==trial['trial_id']],key=lambda e:e['wall_time_ns'])
        actions, snapshots, inferences = matching('vla_first_action'), matching('vla_snapshot'), matching('vla_first_inference')
        if not trial['vla_first_action']:
            assert not actions and not trial['first_policy_chest_frame']
            first_input_proof.append({'trial_id':trial['trial_id'],'first_action_events':0,'retained_without_snapshot':True})
            continue
        assert actions and snapshots and inferences
        action, snapshot, inference = actions[0], snapshots[0], inferences[0]
        chosen = trial['first_policy_chest_frame']
        assert snapshot['path'] == chosen['snapshot_json']
        assert snapshot['skill_id'] == inference['skill_id'] == action['skill_id']
        assert snapshot['generation'] == inference['generation'] == action['generation']
        assert inference['wall_time_ns'] <= action['wall_time_ns']
        metadata = json.loads(Path(snapshot['path']).read_text())
        camera = metadata['cameras']['chest_view']
        assert Path(chosen['rgb']) == Path(snapshot['path']).parent / camera['rgb']
        assert chosen['timestamp_s'] == camera['timestamp_s'] == inference['camera_timestamps']['chest_view']
        first_input_proof.append({'trial_id':trial['trial_id'],'snapshot_event_id':snapshot['event_id'],
                                  'first_inference_event_id':inference['event_id'],'first_action_event_id':action['event_id'],
                                  'same_skill_generation_and_chest_timestamp':True,
                                  'inference_precedes_action_ms':(action['wall_time_ns']-inference['wall_time_ns'])/1e6,
                                  'source_chest_matches_snapshot_metadata':True})
    (ROOT / 'provenance/experiment_first_input_audit.json').write_text(json.dumps(first_input_proof,indent=2)+'\n')
    for sample in samples:
        path = Path(sample['image']); assert path.exists()
        assert sha256(path) == sample['image_sha256'], path
        with Image.open(path) as image:
            assert image.size == tuple(sample['image_size']) == (640, 480)
        source = sample.get('source_video') or sample['source_image']
        assert sha256(source) == sample['source_sha256'], source
        if sample['split'] == 'train':
            assert sample['frame_index'] == 0 and sample['frame_pts'] == 0
    for pose in poses:
        assert pose['annotation']['annotator'] == {'model': 'gpt-6-astra', 'reasoning_effort': 'low'}
        assert len(pose['annotation']['points']) == 4
        assert {p['label'] for p in pose['annotation']['points']} == set('ABCD')
        assert not pose['strict_valid'] or pose['valid']
        assert (ROOT / 'figures/annotations' / f"{pose['sample_id']}.png").is_file()
    reuse_path = ROOT / 'provenance/annotation_reuse.json'
    if reuse_path.exists():
        for entry in json.loads(reuse_path.read_text()):
            assert sha256(ROOT / 'annotations' / entry['file']) == entry['sha256']
            assert sha256(entry['source']) == entry['sha256']
    config=json.loads((ROOT/'analysis_config.json').read_text())
    occlusion_validation=None
    revision=config.get('occlusion_revision')
    if revision:
        previous=Path(revision['previous_revision'])
        old={p['sample_id']:p for p in json.loads((previous/'results/poses.json').read_text())}
        old_config=json.loads((previous/'analysis_config.json').read_text())
        for key in ['geometry_m','pose_estimator','sensitivity','metric']:
            assert config[key]==old_config[key], key
        if not extension:
            assert sha256(ROOT/'preflight.json')==sha256(previous/'preflight.json')
            assert sha256(ROOT/'samples.json')==sha256(previous/'samples.json')
        selected=set(revision['selected_sample_ids'])
        added_points=0
        for pose in poses:
            if pose['sample_id'] in extension.get('added_sample_ids', []):
                continue
            prior=old[pose['sample_id']]
            before={p['label']:p for p in prior['annotation']['points']}
            for point in pose['annotation']['points']:
                old_point=before[point['label']]
                if old_point['u'] is not None and old_point['v'] is not None:
                    assert all(point.get(k)==old_point.get(k) for k in ['u','v','status','uncertainty_px'])
                elif point['u'] is not None:
                    assert pose['sample_id'] in selected and point['status']=='inferred'
                    assert point['uncertainty_px']>0 and point.get('note')
                    added_points+=1
            if pose['sample_id'] not in selected:
                assert pose['annotation']==prior['annotation']
                for key in ['valid','strict_valid','failure_reason','state','reprojection_rms_px','perturbation_states']:
                    assert pose.get(key)==prior.get(key), (pose['sample_id'],key)
            if pose.get('occlusion_estimated'):
                assert pose['sample_id'] in selected and not pose['strict_valid']
        assert {p['sample_id'] for p in poses if p['annotation_source'].get('is_occlusion_estimate')}==selected
        occlusion_validation={'selected_samples':len(selected),'added_inferred_points':added_points,
                              'all_existing_localized_coordinates_statuses_and_radii_unchanged':True,
                              'all_unselected_poses_and_annotations_unchanged':True,
                              'geometry_and_quality_and_metric_rules_unchanged':True,
                              'new_estimates_excluded_from_strict_results':True}
    confirmation_path = ROOT / 'provenance/geometry_confirmation.json'
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text())
        config = json.loads((ROOT / 'analysis_config.json').read_text())
        assert config['geometry_m'] == confirmation['dimensions_m']
        assert pre['geometry']['top_width_m'] == config['geometry_m']['top_width']
        assert pre['geometry']['bottom_width_m'] == config['geometry_m']['bottom_width']
        assert pre['geometry']['side_length_m'] == config['geometry_m']['side']
    all_stats = json.loads((ROOT / 'results/all_training/statistics.json').read_text())
    tail_stats = json.loads((ROOT / 'results/from_00145/statistics.json').read_text())
    assert all_stats['calibration'] == tail_stats['calibration']
    extension_validation = None
    if extension:
        previous = Path(extension['previous_root'])
        prior_poses = {p['sample_id']: p for p in json.loads((previous/'results/poses.json').read_text())}
        current_poses = {p['sample_id']: p for p in poses}
        assert set(current_poses) - set(prior_poses) == set(extension['added_sample_ids'])
        old_pre = json.loads((previous/'preflight.json').read_text())
        assert {t['trial_id'] for t in pre['trials']} - {t['trial_id'] for t in old_pre['trials']} == set(extension['added_trial_ids'])
        assert all_stats['calibration'] == json.loads((previous/'results/calibration.json').read_text())
        plot_settings = json.loads((ROOT/'results/plot_settings.json').read_text())
        prior_settings = json.loads((previous/'results/plot_settings.json').read_text())
        assert plot_settings['covariance_m2'] == prior_settings['covariance_m2']
        for pose in poses:
            if pose['valid']:
                assert plot_settings['axes_x_m'][0] <= pose['state'][0] <= plot_settings['axes_x_m'][1]
                assert plot_settings['axes_normal_distance_m'][0] <= pose['state'][1] <= plot_settings['axes_normal_distance_m'][1]
        for sid, prior in prior_poses.items():
            current = current_poses[sid]
            for key in ['annotation','valid','strict_valid','failure_reason','state','reprojection_rms_px','perturbation_states']:
                assert current.get(key) == prior.get(key), (sid, key)
        for name in ['all_training', 'from_00145']:
            old_stats = json.loads((previous/'results'/name/'statistics.json').read_text())
            current_stats = all_stats if name == 'all_training' else tail_stats
            current_trials = {t['trial_id']: t for t in current_stats['trials']}
            for trial in old_stats['trials']:
                assert {k:current_trials[trial['trial_id']].get(k) for k in trial} == trial, trial['trial_id']
        repeat = json.loads((ROOT/'provenance/new_annotation_comparison.json').read_text())
        assert repeat['sample_count'] == len(extension['added_sample_ids'])
        assert repeat['unresolved_review_sample_ids'] == []
        extension_validation = {'old_250_annotations_and_poses_unchanged': True,
                                'old_trial_distances_and_hits_unchanged': True,
                                'training_calibration_unchanged': True,
                                'training_kde_covariance_unchanged_and_all_valid_points_in_plot': True,
                                'new_trials_match_raw_user_labels_and_corrected_conditions': True,
                                'new_blinded_independent_review_complete': True}
    off_frame_validation = None
    off_frame = config.get('off_frame_revision')
    if off_frame:
        previous = Path(off_frame['previous_root'])
        old_poses = {p['sample_id']:p for p in json.loads((previous/'results/poses.json').read_text())}
        current = {p['sample_id']:p for p in poses}
        assert set(current) == set(old_poses)
        assert sha256(ROOT/'preflight.json') == sha256(previous/'preflight.json')
        assert sha256(ROOT/'samples.json') == sha256(previous/'samples.json')
        assert sha256(ROOT/'provenance/camera_intrinsics.json') == sha256(previous/'provenance/camera_intrinsics.json')
        assert sha256(ROOT/'annotations/off_frame_estimates.jsonl') == off_frame['accepted_annotation_sha256']
        assert sha256(off_frame['accepted_annotation']) == off_frame['accepted_annotation_sha256']
        accepted = {p['sample_id']:p for p in map(json.loads,(ROOT/'annotations/off_frame_estimates.jsonl').read_text().splitlines())}
        for sid, prior in old_poses.items():
            pose = current[sid]
            if sid in off_frame['selected_sample_ids']:
                assert pose['annotation']['points'] == accepted[sid]['points']
                assert pose['off_frame_estimated'] and not pose['strict_valid']
                before = {p['label']:p for p in prior['annotation']['points']}
                for point in pose['annotation']['points']:
                    if point['label'] not in off_frame['approved_points'][sid]:
                        assert point == before[point['label']]
                assert pose['annotation_source']['is_off_frame_estimate']
            else:
                for key in ['annotation','valid','strict_valid','failure_reason','state','reprojection_rms_px','perturbation_states']:
                    assert pose.get(key) == prior.get(key), (sid,key)
        assert all_stats['calibration'] == json.loads((previous/'results/calibration.json').read_text())
        for name, stats in [('all_training',all_stats),('from_00145',tail_stats)]:
            previous_stats = json.loads((previous/'results'/name/'statistics.json').read_text())
            now = {t['trial_id']:t for t in stats['trials']}
            for trial in previous_stats['trials']:
                row = now[trial['trial_id']]
                affected = trial['trial_id'] in off_frame['selected_trial_ids']
                assert row['off_frame_estimated'] == affected
                if not affected:
                    assert {k:row.get(k) for k in trial} == trial, trial['trial_id']
            changed_methods = {now[t]['method'] for t in off_frame['selected_trial_ids']}
            for before in previous_stats['methods']:
                if before['method'] not in changed_methods:
                    assert next(m for m in stats['methods'] if m['method']==before['method']) == before
        off_frame_validation = {'user_accepted_point_file_and_coordinates_exact':True,
                                'unselected_255_annotations_and_poses_unchanged':True,
                                'other_18_trials_and_unaffected_method_statistics_unchanged':True,
                                'cohort_source_images_and_intrinsics_unchanged':True,
                                'training_reference_and_calibration_unchanged':True,
                                'inferred_out_of_frame_point_excluded_from_strict':True,
                                'source':'Single Astra low extrapolation accepted by the user; independent initial annotations did not estimate this missing corner.'}
    for stats in [all_stats, tail_stats]:
        assert stats['reference_pose_valid'] + stats['reference_pose_missing'] == stats['reference_source_frames']
        assert sum(m['n_trials'] for m in stats['methods']) == expected_trials
        for method in stats['methods']:
            assert method['n_pose_valid'] + method['n_pose_missing'] == method['n_trials']
            assert abs(method['neighborhood_hit_rate_pct'] - 100*method['n_neighborhood_hits']/method['n_trials']) < 1e-10
            assert method['strict_neighborhood_hits'] <= method['n_neighborhood_hits']
    for a, b in zip(all_stats['trials'], tail_stats['trials']):
        assert a['trial_id'] == b['trial_id']
        if a['normalized_distance'] is not None:
            assert a['normalized_distance'] <= b['normalized_distance'] + 1e-10
        assert not b['neighborhood_hit'] or a['neighborhood_hit']
    for file in [ROOT / 'annotation_gallery.html', ROOT / 'report.md']:
        content = file.read_text()
        links = re.findall(r'(?:href|src)="([^"#]+)"', content) if file.suffix == '.html' else re.findall(r'\]\(([^)]+)\)', content)
        for link in links:
            if re.match(r'^(https?:|data:|#)', link):
                continue
            target = file.parent / unquote(link.split('#')[0])
            if target == ROOT / 'provenance/final_validation.json':
                continue  # This validation result is written after all checks pass.
            assert target.exists(), (file, link)
    numerical_baseline_path = ROOT / 'provenance/reproduction_numeric_baseline.json'
    numerical_reproduction = None
    if numerical_baseline_path.exists():
        baseline = json.loads(numerical_baseline_path.read_text())
        numerical_reproduction = {name: sha256(ROOT / name) == digest for name,digest in baseline.items()}
        assert all(numerical_reproduction.values()), numerical_reproduction
        (ROOT / 'provenance/numeric_reproduction_check.json').write_text(json.dumps(numerical_reproduction, indent=2)+'\n')
    manifest = []
    for folder in ['annotations', 'figures', 'results', 'pipeline', 'provenance', 'jobs']:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in str(path) and path.name not in ['artifact_manifest.json','final_validation.json','reproduction_log.txt']:
                manifest.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    for name in ['report.md', 'README.md', 'analysis_config.json', 'samples.json', 'preflight.json', 'annotation_protocol.md', 'annotation_gallery.html']:
        path = ROOT / name
        manifest.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    (ROOT / 'provenance/artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = {'status': 'passed', 'source_video_hashes_checked': 238, 'source_experiment_image_hashes_checked': expected_experiments,
              'first_frame_png_hashes_checked': len(samples), 'annotation_model_provenance_checked': len(poses),
              'formal_trials': expected_trials, 'vla_started': expected_started, 'reference_subset_and_shared_calibration': True,
              'metric_denominators_and_nested_reference_monotonicity': True, 'local_report_and_gallery_links': 'all exist',
              'artifact_files_hashed': len(manifest), 'scope': 'File integrity and software consistency; no independent physical measurement validation.'}
    result['existing_annotations_reused_without_changes'] = reuse_path.exists()
    result['occlusion_annotation_layer_validation']=occlusion_validation
    result['geometry_matches_user_confirmation'] = confirmation_path.exists()
    result['numeric_outputs_reproduced'] = numerical_reproduction
    result['all_first_policy_inputs_verified_against_raw_events'] = True
    result['cohort_extension_validation'] = extension_validation
    result['off_frame_validation'] = off_frame_validation
    (ROOT / 'provenance/final_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
