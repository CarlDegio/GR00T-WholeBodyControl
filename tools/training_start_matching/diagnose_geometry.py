"""Audit the recovered upward coordinate after the user's dimension correction."""
from pathlib import Path
import json
import numpy as np
from paths import ROOT


def summarize(poses):
    groups = {
        'all_training': [p for p in poses if p['split'] == 'train' and p['valid']],
        'from_00145': [p for p in poses if p['split'] == 'train' and p['valid'] and p['episode_id'] >= 145],
        'experiments': [p for p in poses if p['split'] == 'experiment' and p['valid']],
    }
    result = {}
    for name, rows in groups.items():
        values = np.array([p['camera_position_object_m'][1] for p in rows])
        result[name] = {'n_solved': len(rows), 'n_negative_wall_y': int((values < 0).sum()),
                        'median_wall_y_m': float(np.median(values)) if len(values) else None,
                        'range_wall_y_m': [float(values.min()), float(values.max())] if len(values) else None,
                        'sample_ids_negative_wall_y': [p['sample_id'] for p in rows if p['camera_position_object_m'][1] < 0]}
    return result


def main():
    poses = json.loads((ROOT / 'results/poses.json').read_text())
    config = json.loads((ROOT / 'analysis_config.json').read_text())
    confirmation_path = ROOT / 'provenance/geometry_confirmation.json'
    confirmation = json.loads(confirmation_path.read_text()) if confirmation_path.exists() else None
    counts = summarize(poses)
    clear = all(v['n_negative_wall_y'] == 0 for v in counts.values())
    record = {
        'status': 'dimensions_confirmed_sign_check_clear' if confirmation and clear else 'geometry_consistency_needs_review',
        'diagnostic': counts,
        'geometry_m': config['geometry_m'],
        'user_confirmation': confirmation,
        'pixel_annotations_changed': bool(config.get('occlusion_revision')) or (bool(confirmation.get('pixel_annotations_changed', False)) if confirmation else None),
        'existing_localized_pixels_changed': False,
        'occlusion_estimated_samples': sum(p.get('occlusion_estimated',False) for p in poses),
        'positive_wall_y_imposed_as_selection_constraint': False,
        'interpretation': 'Counts describe unconstrained minimum-reprojection-error solutions after updating the measured trapezoid. A positive wall-upward coordinate resolves the previous sign concern, but does not independently validate metric accuracy.',
        'reporting_rule': 'Report model-based first-frame camera geometry, observable reference counts, missingness and uncertainty. Independent physical pose accuracy and definitive method superiority are not established.',
        'state_omissions': 'The primary distance uses X, Z and optical-forward heading. Wall-frame Y and pitch are saved but omitted; a neighborhood hit does not imply full visual-observation equivalence.',
        'independent_metric_space_validation': False,
    }
    if confirmation and confirmation.get('previous_analysis_root'):
        previous_path = Path(confirmation['previous_analysis_root']) / 'results/poses.json'
        if previous_path.exists():
            record['previous_geometry_diagnostic'] = summarize(json.loads(previous_path.read_text()))
    (ROOT / 'results/geometry_diagnostics.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'geometry_status': record['status'], 'negative_wall_y': {k: [v['n_negative_wall_y'], v['n_solved']] for k, v in counts.items()}}))


if __name__ == '__main__':
    main()
