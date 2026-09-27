"""Compare two Astra-annotated upper-edge definitions on four diagnostic images."""
import json
from paths import ROOT
import numpy as np
import matplotlib.pyplot as plt
from estimate import object_points, solve_candidates
from present import draw_annotation, save


def main():
    path = ROOT / 'annotations/rim_correspondence_diagnostic.jsonl'
    if not path.exists():
        print('No separate Astra rim annotations; diagnostic comparison skipped.')
        return
    annotations = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    poses = {p['sample_id']: p for p in json.loads((ROOT / 'results/poses.json').read_text())}
    config = json.loads((ROOT / 'analysis_config.json').read_text())
    cal = json.loads((ROOT / 'provenance/camera_intrinsics.json').read_text())['streams']['chest_view']
    K = np.array([[cal['fx'], 0, cal['cx']], [0, cal['fy'], cal['cy']], [0, 0, 1.]])
    distortion = np.array(cal['distortion_coeffs'])
    records = []
    fig, axes = plt.subplots(len(annotations), 2, figsize=(12, 4.5*len(annotations)))
    for row, a in enumerate(annotations):
        assert a['annotator'] == {'model': 'gpt-6-astra', 'reasoning_effort': 'low'}
        original = poses[a['sample_id']]
        alternate = dict(original, annotation=a)
        points = np.array([[p['u'], p['v']] for p in a['points']], float)
        solutions = solve_candidates(points, object_points(config), K, distortion)
        best = solutions[0] if solutions else None
        record = {'sample_id': a['sample_id'], 'body_camera_wall_y_m': original['camera_position_object_m'][1],
                  'body_reprojection_rms_px': original['reprojection_rms_px'], 'rim_pose_candidates': solutions,
                  'rim_camera_wall_y_m': best['camera_position_object_m'][1] if best else None,
                  'rim_reprojection_rms_px': best['reprojection_rms_px'] if best else None,
                  'used_in_primary_metrics': False}
        records.append(record)
        def pair(pair_axes):
            draw_annotation(pair_axes[0], original)
            draw_annotation(pair_axes[1], alternate)
            pair_axes[0].set_title(f"{a['sample_id']} · A/B below projecting rim\nSame 29/23/9 cm model: wall Y={record['body_camera_wall_y_m']:+.3f} m", fontsize=10)
            pair_axes[1].set_title(f"{a['sample_id']} · A/B at OUTER projecting rim\nSame model: wall Y={record['rim_camera_wall_y_m']:+.3f} m", fontsize=10)
        pair(axes[row])
        if a['sample_id'] == 'img_0216':
            single, pair_axes = plt.subplots(1, 2, figsize=(12, 5))
            pair(pair_axes)
            single.suptitle('Which physical edge did the 29 cm measurement use?', fontsize=16)
            single.text(.05, .01, 'Both point sets annotated by Astra low. Right-side definition is diagnostic only; physical dimensions still require confirmation.', fontsize=9)
            single.tight_layout(rect=(0, .05, 1, .9))
            save(single, ROOT / 'figures/rim_correspondence_example', vector=False)
    fig.suptitle('Body corners vs. outer-rim corners: physical correspondence diagnostic', fontsize=16, y=.999)
    fig.tight_layout(rect=(0, 0, 1, .975))
    save(fig, ROOT / 'figures/rim_correspondence_comparison', vector=False)
    result = {'purpose': 'Diagnose whether measured upper width belongs to the projecting rim or body. No primary annotations or matching results replaced.',
              'dimensions_confirmed_for_outer_rim': False, 'not_an_independent_physical_validation': True,
              'records': records}
    (ROOT / 'results/rim_correspondence_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'diagnostic_images': len(records), 'outer_rim_positive_wall_y': sum(r['rim_camera_wall_y_m'] > 0 for r in records), 'primary_results_changed': False}))


if __name__ == '__main__':
    main()
