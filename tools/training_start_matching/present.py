"""Scientific figures and a local, auditable annotation/image-pair gallery."""
from pathlib import Path
from paths import ROOT
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image

from metrics import METHODS, wrap_deg


FIG = ROOT / 'figures'
NAMES = {'full_vln': 'Full VLN', 'nav_direct_vla': 'Nav → VLA',
         'geometric_vla': 'Geometric → VLA', 'full_objectnav': 'Full ObjectNav'}
COLORS = dict(zip(METHODS, ['#7543a4', '#c24942', '#d08b1f', '#167663']))
REFERENCES = [('all_training', 'All training episodes'), ('from_00145', 'Training episodes ≥ 00145')]
POINT_COLORS = {'A': '#ffdf42', 'B': '#ff6d62', 'C': '#68e3ff', 'D': '#9eff74'}
CONFIG = json.loads((ROOT / 'analysis_config.json').read_text())
NEW_TRIAL_IDS = set(CONFIG.get('cohort_extension', {}).get('added_trial_ids', []))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, path, vector=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path = ROOT / 'results/geometry_diagnostics.json'
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic['status'] != 'dimensions_confirmed_sign_check_clear':
            config=json.loads((ROOT/'analysis_config.json').read_text())
            warning='DIAGNOSTIC: geometry consistency needs review'
            if config.get('occlusion_revision'):
                reference = 'from_00145' if 'from_00145' in path.parts else 'all_training'
                n=diagnostic['diagnostic'][reference]['n_negative_wall_y']
                experiment_n=diagnostic['diagnostic']['experiments']['n_negative_wall_y']
                warning = f'Exploratory estimates: {n} training / {experiment_n} experimental poses have wall-Y sign warnings' if n or experiment_n else None
            if warning:
                fig.text(.995, 1.005, warning,
                         ha='right', va='bottom', color='#9b3c27', fontsize=9)
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight', facecolor='white')
    if vector:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def case_label(p):
    label = p['case_id'].removeprefix('visible_')
    return label + ('*' if p.get('trial_id') in NEW_TRIAL_IDS else '') + ('†' if p.get('off_frame_estimated') else '')


def pose_marker(p):
    return 'D' if p.get('inferred_corners') else 'o'


def label_experiments(ax, poses, heading=False):
    """Place case labels without moving measured points or obscuring nearby labels."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    pixel_scale = ax.figure.dpi / 72
    xy = [(p['state'][1], p['state'][2]) if heading else tuple(p['state'][:2]) for p in poses]
    centers = [ax.transData.transform(point) for point in xy]
    marker_boxes = [Bbox.from_bounds(x-7*pixel_scale, y-7*pixel_scale, 14*pixel_scale, 14*pixel_scale)
                    for x, y in centers]
    occupied = []
    for p, point in zip(poses, xy):
        color = COLORS[p['method']]
        label = ax.annotate(case_label(p), point, xytext=(10, 10), textcoords='offset points',
                            fontsize=10, color=color, weight='bold', zorder=8)
        choices = []
        for radius in (10, 20, 32, 46, 62):
            for dx, dy in [(1,1),(-1,1),(1,-1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)]:
                offset = (radius*dx, radius*dy)
                label.set_position(offset)
                label.set_ha('right' if dx < 0 else 'left' if dx > 0 else 'center')
                label.set_va('top' if dy < 0 else 'bottom' if dy > 0 else 'center')
                box = label.get_window_extent(renderer).expanded(1.15, 1.2)
                intersections = sum(box.overlaps(other) for other in marker_boxes + occupied)
                inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                score = intersections*10000 + (0 if inside else 100000) + np.hypot(*offset)
                choices.append((score, offset, label.get_ha(), label.get_va(), box))
        _, offset, ha, va, box = min(choices, key=lambda item: item[0])
        label.remove()
        occupied.append(box)
        ax.annotate(case_label(p), point, xytext=offset, textcoords='offset points',
                    ha=ha, va=va, fontsize=10, color=color, weight='bold', zorder=8,
                    arrowprops=dict(arrowstyle='-', color=color, lw=.7, shrinkA=3, shrinkB=7))


def draw_annotation(ax, pose, circles=True):
    ax.imshow(Image.open(pose['image']))
    canvas_height = 600 if pose.get('off_frame_estimated') else 480
    if pose.get('off_frame_estimated'):
        ax.axhspan(479.5, canvas_height, color='#edf2f6', zorder=0)
        ax.axhline(479.5, color='#627b8e', linestyle='--', linewidth=.8)
        ax.text(12, 542, 'Blank extension; original pixel coordinates unchanged', fontsize=9, color='#344d61')
    points = pose['annotation']['points']
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        if p['u'] is None or q['u'] is None:
            continue
        inferred = p['status'] != 'visible' or q['status'] != 'visible'
        ax.plot([p['u'], q['u']], [p['v'], q['v']], color='#ffe56c',
                linewidth=1.2, linestyle='--' if inferred else '-')
    for p in points:
        if p['u'] is None:
            continue
        color = '#d52b83' if p.get('out_of_frame') else POINT_COLORS[p['label']]
        ax.scatter(p['u'], p['v'], s=24, edgecolor=color,
                   facecolor=color if p['status'] == 'visible' else 'none', linewidth=1.4)
        if circles and p.get('uncertainty_px'):
            ax.add_patch(Circle((p['u'], p['v']), p['uncertainty_px'], fill=False,
                                color=color, linewidth=.7, alpha=.7))
        ax.annotate(p['label'] + ('?' if p['status'] != 'visible' else ''),
                    (p['u'], p['v']), xytext=(4, -10), textcoords='offset points',
                    color=color, fontsize=9, weight='bold',
                    bbox={'facecolor': 'black', 'alpha': .5, 'pad': .4, 'edgecolor': 'none'})
    ax.set_xlim(0, 640); ax.set_ylim(canvas_height, 0); ax.axis('off')


def annotation_gallery(poses):
    folder = FIG / 'annotations'; folder.mkdir(parents=True, exist_ok=True)
    cards = []
    for p in poses:
        fig, ax = plt.subplots(figsize=(6.4, 6.0 if p.get('off_frame_estimated') else 4.8)); draw_annotation(ax, p)
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(folder / f"{p['sample_id']}.png", dpi=100, facecolor='white')
        plt.close(fig)
        label = f"episode {p['episode_id']:06d}" if p['split'] == 'train' else f"{NAMES[p['method']]} {case_label(p)}"
        status = 'strict' if p['strict_valid'] else ('candidate' if p['valid'] else 'missing')
        pt_text = '; '.join(f"{x['label']}: {x['status']} ({x['u']}, {x['v']}); {x.get('note', '')}" for x in p['annotation']['points'])
        detail = p['failure_reason'] or f"RMS {p['reprojection_rms_px']:.2f}px; inferred={p['inferred_corners']}; ambiguous={p['ambiguous']}; stable={p['stable']}"
        source = p['annotation_source']; source_name = Path(source['file']).name
        cards.append(f'''<article data-split="{p['split']}" data-status="{status}"><h3>{html.escape(label)} · {p['sample_id']}</h3>
<a href="figures/annotations/{p['sample_id']}.png"><img loading="lazy" src="figures/annotations/{p['sample_id']}.png" alt="{html.escape(label)} four corners"></a>
<p><b>{status}</b> · {html.escape(detail)}</p><p>{html.escape(p['annotation'].get('notes',''))}</p>
<details><summary>逐点记录及来源</summary><p>{html.escape(pt_text)}</p><a href="annotations/{source_name}">{source_name}:{source['line']}</a></details>
<a href="frames/{p['sample_id']}.png">原始首帧</a></article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>篮子前壁四点标注复核</title><style>body{font:15px/1.6 system-ui;margin:24px;background:#f5f6f8;color:#18202d}h1{font-size:25px}a{color:#175b9b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}article{background:white;padding:14px;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}h3{font-size:16px;margin:0 0 10px}p{margin:6px 0}select{font-size:16px;margin:8px 16px 20px 0}details{font-size:13px}article[hidden]{display:none}</style>
<h1>250 张首帧：Astra low 四点标注与几何质检</h1><p>A 左上，B 右上，C 右下，D 左下；实心点为可见角点，空心点 / 虚线 / ? 为推断。小圆是标注者给出的像素扰动半径。原图像素独立保留。</p>
<p>strict：四点可见、无检出的分支歧义且通过扰动稳定性检查；candidate：几何检查通过但不满足 strict；missing：四点不足或几何检查失败。此状态不代表操作成功与否。</p>
<p><a href="report.md">完整报告</a> · <a href="results/poses.csv">位姿 CSV</a> · <a href="annotation_protocol.md">标注协议</a></p>
<label>样本 <select id="split"><option value="all">全部</option><option value="experiment">实验</option><option value="train">训练</option></select></label>
<label>质检 <select id="status"><option value="all">全部</option><option value="strict">strict</option><option value="candidate">candidate</option><option value="missing">missing</option></select></label><span id="count"></span><main class="grid">'''
    document += '\n'.join(cards) + '''</main><script>function filter(){let n=0;document.querySelectorAll('article').forEach(x=>{x.hidden=!((split.value==='all'||x.dataset.split===split.value)&&(statusSelect.value==='all'||x.dataset.status===statusSelect.value));if(!x.hidden)n++});document.getElementById('count').textContent=n+' 张'}const split=document.getElementById('split'),statusSelect=document.getElementById('status');split.onchange=statusSelect.onchange=filter;filter()</script></html>'''
    document = document.replace('250 张首帧', f'{len(poses)} 张首帧')
    (ROOT / 'annotation_gallery.html').write_text(document)
    experiments = [p for p in poses if p['split'] == 'experiment']
    columns = max(sum(p['method'] == method for p in experiments) for method in METHODS)
    fig, axes = plt.subplots(4, columns, figsize=(5 * columns, 16), squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for row, method in enumerate(METHODS):
        for ax, p in zip(axes[row], sorted([p for p in experiments if p['method'] == method], key=case_label)):
            draw_annotation(ax, p, circles=False)
            status = 'strict' if p['strict_valid'] else ('candidate' if p['valid'] else 'unresolved')
            ax.set_title(f"{NAMES[method]} · {case_label(p)} · {status}", fontsize=11)
    fig.suptitle('Actual first VLA chest inputs — reviewed front-wall corners', fontsize=17, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .98)); save(fig, FIG / 'experiment_annotations', vector=False)


def fixed_kde(points, grid, covariance):
    inverse = np.linalg.inv(covariance)
    out = np.zeros(len(grid))
    for point in points:
        delta = grid - point
        out += np.exp(-.5 * np.einsum('ni,ij,nj->n', delta, inverse, delta))
    return out / (len(points) * 2 * np.pi * np.sqrt(np.linalg.det(covariance)))


def density_figures(poses, summaries):
    config = json.loads((ROOT / 'analysis_config.json').read_text())
    with_occlusion = bool(config.get('occlusion_revision'))
    training = [p for p in poses if p['split'] == 'train' and p['valid']]
    experiments = [p for p in poses if p['split'] == 'experiment' and p['valid']]
    all_xz = np.array([p['state'][:2] for p in training + experiments])
    mins = all_xz.min(axis=0); maxs = all_xz.max(axis=0)
    xmin, xmax = mins[0] - .12, maxs[0] + .12
    zmin, zmax = 0., maxs[1] + .13
    xx, zz = np.meshgrid(np.linspace(xmin, xmax, 320), np.linspace(zmin, zmax, 220))
    grid = np.c_[xx.ravel(), zz.ravel()]
    training_xz = np.array([p['state'][:2] for p in training])
    covariance = np.cov(training_xz.T) * len(training) ** (-1 / 3) + np.eye(2) * 1e-8
    densities = {}; references = {}
    for name, _ in REFERENCES:
        references[name] = [p for p in training if name == 'all_training' or p['episode_id'] >= 145]
        densities[name] = fixed_kde(np.array([p['state'][:2] for p in references[name]]), grid, covariance).reshape(xx.shape)
    vmax = max(float(x.max()) for x in densities.values())
    norm = Normalize(0, vmax)
    settings = {'covariance_m2': covariance.tolist(), 'bandwidth_rule': 'Scott n^(-1/6) in 2D, covariance fit once on all valid training poses',
                'density_normalized_per_reference': True, 'axes_x_m': [xmin, xmax], 'axes_normal_distance_m': [zmin, zmax], 'common_color_max_m_minus2': vmax,
                'arrows': 'Camera optical-forward projected onto X-Z plane; fixed display length 0.105 m, not a displacement',
                'no_nominal_valid_points_outside_plot': True}
    (ROOT / 'results/plot_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    for name, title in REFERENCES:
        ref = references[name]; summary = summaries[name]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        fig.subplots_adjust(left=.07, right=.87, bottom=.14, top=.85, hspace=.3, wspace=.12)
        states = np.array([p['state'] for p in ref])
        for ax, method in zip(axes.flat, METHODS):
            ax.pcolormesh(xx, zz, densities[name], cmap='Blues', norm=norm, shading='auto', rasterized=True)
            old_states = np.array([p['state'] for p in ref if not p.get('occlusion_estimated')]).reshape(-1,3)
            added_states = np.array([p['state'] for p in ref if p.get('occlusion_estimated')]).reshape(-1,3)
            ax.scatter(old_states[:, 0], old_states[:, 1], s=8, c='#4b5660', alpha=.48, linewidths=0, zorder=3)
            if len(added_states):
                ax.scatter(added_states[:, 0], added_states[:, 1], s=23, marker='D',
                           facecolors='none', edgecolors='#727b83', linewidths=.8, zorder=4)
            flagged=np.array([p['state'] for p in ref if p['camera_position_object_m'][1]<0]).reshape(-1,3)
            if with_occlusion and len(flagged):
                ax.scatter(flagged[:,0],flagged[:,1],s=42,marker='x',c='#bb4b36',linewidths=1.5,zorder=5)
            half_width=config['geometry_m']['top_width']/2
            ax.plot([-half_width, half_width], [0, 0], color='#26364c', linewidth=5, solid_capstyle='butt', clip_on=False)
            for p in [p for p in experiments if p['method'] == method]:
                x, z, angle = p['state']; color = COLORS[method]
                ax.scatter(x, z, s=110, marker=pose_marker(p), facecolors=color if p['strict_valid'] else 'white',
                           edgecolors=color, linewidths=2, zorder=7)
                ax.arrow(x, z, .105 * np.sin(np.radians(angle)), -.105 * np.cos(np.radians(angle)),
                         color=color, width=.003, head_width=.025, length_includes_head=True, zorder=6)
            stat = next(m for m in summary['methods'] if m['method'] == method)
            ax.set_title(f"{NAMES[method]}    pose {stat['n_pose_valid']}/{stat['n_trials']} · hits {stat['n_neighborhood_hits']}/{stat['n_trials']}",
                         fontsize=12, loc='left', pad=9)
            ax.set_xlim(xmin, xmax); ax.set_ylim(zmin, zmax); ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.12)
            label_experiments(ax, [p for p in experiments if p['method'] == method])
        for ax in axes[1]: ax.set_xlabel('Camera lateral position X in front-wall frame (m)')
        for ax in axes[:, 0]: ax.set_ylabel('Camera distance along outward wall normal Z (m)')
        cax = fig.add_axes([.9, .27, .018, .43])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Blues'), cax=cax)
        bar.set_label('Training KDE density (m⁻²)')
        legend = [Line2D([], [], marker='o', linestyle='', color='#4b5660', markersize=4, label='Training pose'),
                  Line2D([], [], marker='o', linestyle='', color='#555', markerfacecolor='white', markersize=8, label='Candidate experiment'),
                  Line2D([], [], marker='o', linestyle='', color='#555', markersize=8, label='Strict experiment')]
        if with_occlusion:
            legend.extend([Line2D([], [], marker='D', linestyle='', color='#727b83', markerfacecolor='white', markersize=5, label='Added training estimate'),
                           Line2D([], [], marker='D', linestyle='', color='#555', markerfacecolor='white', markersize=8, label='Experiment with inferred corners')])
            if any(p['camera_position_object_m'][1]<0 for p in ref):
                legend.append(Line2D([],[],marker='x',linestyle='',color='#bb4b36',markersize=7,label='Training Y-sign warning'))
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.47, .04), ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f'{title}: chest-camera start geometry', fontsize=18, x=.47, y=.97)
        fig.text(.07, .905, f"{summary['reference_pose_valid']}/{summary['reference_source_frames']} training first frames estimable; common axes, bandwidth and color scale. Arrows show viewing direction.", fontsize=11)
        new_note = '  * = new formal trial (September 10).' if NEW_TRIAL_IDS else ''
        if config.get('off_frame_revision'):
            new_note += '  † = confirmed off-frame estimate.'
        fig.text(.07, .018, 'Camera-to-wall coordinates; training KDE with experimental points. Counts include missing poses.' + new_note, fontsize=9, color='#4b5660')
        save(fig, FIG / name / 'position_density')

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True, sharey=True)
        heading_min = min(p['state'][2] for p in training + experiments) - 10
        heading_max = max(p['state'][2] for p in training + experiments) + 10
        for ax, method in zip(axes.flat, METHODS):
            ax.scatter(states[:, 1], states[:, 2], c='#768693', s=13, alpha=.55, linewidths=0)
            for p in [p for p in experiments if p['method'] == method]:
                _, z, angle = p['state']; color = COLORS[method]
                ax.scatter(z, angle, s=100, marker=pose_marker(p), facecolors=color if p['strict_valid'] else 'white', edgecolors=color, linewidths=2)
            ax.set_title(NAMES[method], loc='left'); ax.set_xlim(0, zmax); ax.set_ylim(heading_min, heading_max); ax.grid(alpha=.2)
        for ax in axes[1]: ax.set_xlabel('Camera wall-normal distance Z (m)')
        for ax in axes[:, 0]: ax.set_ylabel('Optical-forward heading around wall Y (deg)')
        fig.suptitle(f'{title}: viewing direction and distance', fontsize=17)
        fig.text(.08, .015, '0° faces the wall; positive angles turn toward +X. Filled: strict. Diamonds: inferred corners. Not robot yaw.' + new_note, fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .95))
        for ax, method in zip(axes.flat, METHODS):
            label_experiments(ax, [p for p in experiments if p['method'] == method], heading=True)
        save(fig, FIG / name / 'heading_distance')


def score_figure(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharey=True)
    for ax, (name, title) in zip(axes, REFERENCES):
        stats = summaries[name]['methods']; x = np.arange(4)
        values = np.array([s['neighborhood_hit_rate_pct'] for s in stats])
        intervals = np.array([s['hit_rate_wilson95_pct'] for s in stats])
        ax.bar(x, values, width=.55, color=[COLORS[m] for m in METHODS], alpha=.8)
        ax.errorbar(x, values, yerr=np.maximum(0, np.array([values - intervals[:, 0], intervals[:, 1] - values])),
                    fmt='none', capsize=4, color='#333', linewidth=1.3)
        ax.scatter(x, [s['strict_hit_rate_pct'] for s in stats], marker='_', s=260, color='black', linewidths=2.5, zorder=5)
        for i, s in enumerate(stats):
            ax.text(i, 110, f"{s['n_neighborhood_hits']}/{s['n_trials']}\npose {s['n_pose_valid']}/{s['n_trials']}", ha='center', va='center', fontsize=10)
        ax.set_ylim(-5, 122); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xticks(x, [NAMES[m].replace(' → ', '\n→ ') for m in METHODS], fontsize=10)
        ax.set_title(title, fontsize=14, pad=14); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    axes[0].set_ylabel('Documented training-neighborhood hits / all formal trials (%)')
    fig.suptitle('Training-neighborhood coverage: exploratory, first-frame geometry', fontsize=17)
    fig.text(.05, .035, 'Bars: primary candidate estimates. Black ticks: strict estimates + strict training reference. Whiskers: descriptive Wilson 95% intervals.\nMissing geometry stays in the trial denominator; a missing pose does not establish a physical mismatch. No definite ranking with this sample size.', fontsize=10)
    fig.tight_layout(rect=(0, .13, 1, .94)); save(fig, FIG / 'coverage_comparison')


def uncertainty_figure(poses):
    candidates = [p for p in poses if p['split'] == 'experiment' and p['valid']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for i, p in enumerate(candidates):
        color = COLORS[p['method']]; label = f"{NAMES[p['method']]} {case_label(p)}"
        axes[0].barh(i, 100*p.get('perturbation_xz_p95_radius_m', np.nan), color=color, alpha=.7)
        axes[1].barh(i, p.get('perturbation_heading_p95_deviation_deg', np.nan), color=color, alpha=.7)
        if p['ambiguous']: label += ' [two branches]'
        axes[0].text(-.03, i, label, ha='right', va='center', transform=axes[0].get_yaxis_transform(), fontsize=10)
    for ax in axes:
        ax.set_yticks([]); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    axes[0].axvline(15, color='#222', linestyle='--', linewidth=1); axes[0].set_xlabel('95th-percentile X-Z displacement (cm)')
    axes[1].axvline(15, color='#222', linestyle='--', linewidth=1); axes[1].set_xlabel('95th-percentile heading displacement (deg)')
    fig.suptitle('Sensitivity to annotated-corner pixel perturbations', fontsize=17)
    fig.text(.04, .025, '128 uniform-disk perturbations per image using the annotator-provided radii. Conditional on the nominal solution branch.\nThese are subjective robustness summaries, not physical error bounds or statistical confidence intervals. Dashed lines mark strict stability cutoffs.', fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .94)); save(fig, FIG / 'pixel_sensitivity')


def nearest_pairs(poses, summaries):
    by_trial = {p['trial_id']: p for p in poses if p['split'] == 'experiment'}
    by_id = {p['sample_id']: p for p in poses}
    for name, title in REFERENCES:
        neighbors = json.loads((ROOT / 'results' / name / 'nearest_neighbors.json').read_text())
        neighbor_map = {n['trial_id']: n for n in neighbors}
        for method in METHODS:
            trials = [t for t in summaries[name]['trials'] if t['method'] == method]
            fig, axes = plt.subplots(len(trials), 2, figsize=(11, 4.35*len(trials)), squeeze=False)
            for row, t in enumerate(trials):
                pose = by_trial.get(t['trial_id'])
                if pose:
                    axes[row, 0].imshow(Image.open(pose['image']))
                    axes[row, 0].set_title(f"Experiment {case_label(pose)} · {t['trial_id']}", fontsize=10)
                else:
                    axes[row, 0].text(.5, .5, 'Formal trial did not start VLA\nNo policy-input chest frame', ha='center', va='center', transform=axes[row, 0].transAxes)
                    axes[row, 0].set_title(t['trial_id'], fontsize=10)
                match = neighbor_map.get(t['trial_id'])
                if match:
                    best = match['neighbors'][0]; target = by_id[best['sample_id']]
                    axes[row, 1].imshow(Image.open(target['image']))
                    axes[row, 1].set_title(f"Training {best['episode_id']:06d} · D={best['distance']:.3f} · {'hit' if t['neighborhood_hit'] else 'outside'}\nΔXZ={100*t['nearest_position_distance_m']:.1f} cm; Δheading={t['nearest_heading_difference_deg']:.1f}°", fontsize=10)
                else:
                    axes[row, 1].text(.5, .5, 'No geometric nearest-neighbor assignment\n' + t['failure_reason'], ha='center', va='center', fontsize=10, transform=axes[row, 1].transAxes)
                for ax in axes[row]: ax.axis('off')
            fig.suptitle(f'{NAMES[method]} · {title}\nNearest neighbor in scaled [X, Z, heading]; source images without overlays', fontsize=14, y=.999)
            fig.tight_layout(rect=(0, 0, 1, .97)); save(fig, FIG / name / f'nearest_{method}', vector=False)


def occlusion_review(poses):
    added = sorted([p for p in poses if p['split']=='experiment' and p.get('occlusion_estimated')],
                   key=lambda p:(METHODS.index(p['method']),p['case_id']))
    if not added:
        return
    fig, axes = plt.subplots(int(np.ceil(len(added)/2)), 2, figsize=(13, 5.2*int(np.ceil(len(added)/2))), squeeze=False)
    for ax, p in zip(axes.flat, added):
        draw_annotation(ax,p)
        points=p['annotation']['points']
        uv=np.array([[q['u'],q['v']] for q in points if q['u'] is not None])
        low=np.maximum(uv.min(axis=0)-[55,55],[0,0]); high=np.minimum(uv.max(axis=0)+[55,40],[640,480])
        ax.set_xlim(low[0],high[0]); ax.set_ylim(high[1],low[1]); ax.set_aspect('equal')
        estimates='; '.join(f"{q['label']}=({q['u']},{q['v']}), r={q['uncertainty_px']}px" for q in points if q['label'] in p['occlusion_estimated_labels'])
        status=f"RMS {p.get('reprojection_rms_px',float('nan')):.2f}px" if p['valid'] else p['failure_reason']
        ax.set_title(f"{NAMES[p['method']]} {case_label(p)} · {p['sample_id']}\n{estimates} · {status}",fontsize=11,pad=12)
    for ax in list(axes.flat)[len(added):]:ax.axis('off')
    fig.suptitle('Astra low: bottle-occluded corner estimates',fontsize=17)
    fig.text(.04,.02,'Crops of original first frames. Hollow corners, dashed edges and circles show inferred points and subjective pixel radii.',fontsize=10)
    fig.subplots_adjust(left=.025,right=.975,bottom=.09,top=.85,hspace=.5,wspace=.06)
    save(fig,FIG/'occlusion_estimate_review')


def main():
    poses = json.loads((ROOT / 'results/poses.json').read_text())
    summaries = {name: json.loads((ROOT / 'results' / name / 'statistics.json').read_text()) for name, _ in REFERENCES}
    FIG.mkdir(exist_ok=True)
    annotation_gallery(poses)
    density_figures(poses, summaries)
    score_figure(summaries)
    uncertainty_figure(poses)
    nearest_pairs(poses, summaries)
    occlusion_review(poses)
    print(json.dumps({'figures_png': len(list(FIG.rglob('*.png'))), 'review_images': len(poses), 'gallery': str(ROOT / 'annotation_gallery.html')}))


if __name__ == '__main__':
    main()
