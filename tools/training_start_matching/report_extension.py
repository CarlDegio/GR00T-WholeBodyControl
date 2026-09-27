"""Report a frozen trial-cohort extension without changing the training reference."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import json

import numpy as np

from paths import ROOT
from metrics import METHODS, wilson

NAMES = {'full_vln': 'Full VLN', 'nav_direct_vla': 'Nav → VLA',
         'geometric_vla': 'Geometric → VLA', 'full_objectnav': 'Full ObjectNav'}


def fraction(k, n):
    return f'{k}/{n}（{100*k/n:.1f}%）' if n else '不适用'


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def generate(config, poses, summaries, diagnostic, made):
    extension = config['cohort_extension']
    off_frame = config.get('off_frame_revision')
    previous = Path(off_frame['previous_root'] if off_frame else extension['previous_root'])
    pre = json.loads((ROOT/'preflight.json').read_text())
    new_ids = set(extension['added_trial_ids'])
    new_trials = sorted([t for t in pre['trials'] if t['trial_id'] in new_ids],
                        key=lambda t: t['trial_start_wall_time_ns'])
    all_s, tail_s = summaries['all_training'], summaries['from_00145']
    by_reference = {name: {t['trial_id']: t for t in summary['trials']}
                    for name, summary in summaries.items()}
    methods = {name: {m['method']: m for m in summary['methods']}
               for name, summary in summaries.items()}
    old_stats = {name: json.loads((previous/'results'/name/'statistics.json').read_text())
                 for name in summaries}
    by_pose = {p['trial_id']: p for p in poses if p['split'] == 'experiment'}
    new_poses = [p for p in poses if p.get('trial_id') in new_ids]
    review = json.loads((ROOT/'provenance/new_annotation_comparison.json').read_text())
    differences = [p['distance_px'] for s in review['samples'] for p in s['points']
                   if p['distance_px'] is not None]
    n_trials = len(pre['trials'])
    n_started = sum(t['vla_first_action'] for t in pre['trials'])
    n_pose = sum(p['split'] == 'experiment' and p['valid'] for p in poses)
    n_strict = sum(p['split'] == 'experiment' and p['strict_valid'] for p in poses)
    fresh_results = []
    for t in new_trials:
        a, b = (by_reference[name][t['trial_id']] for name in ['all_training', 'from_00145'])
        fresh_results.append({
            'trial_id': t['trial_id'], 'method': t['method'], 'case_id': t['case_id'],
            'start_time': datetime.fromtimestamp(t['trial_start_wall_time_ns']/1e9, ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'independent_success': t['annotation_values'].get('success'),
            'pose_valid': a['pose_valid'], 'strict_pose_valid': a['strict_pose_valid'],
            'failure_reason': a['failure_reason'], 'all_training_distance': a['normalized_distance'],
            'all_training_hit': a['neighborhood_hit'], 'all_training_nearest_episode': a['nearest_episode_id'],
            'from_00145_distance': b['normalized_distance'], 'from_00145_hit': b['neighborhood_hit'],
            'from_00145_nearest_episode': b['nearest_episode_id'],
        })
    new_methods = []
    for method in METHODS:
        rows = [r for r in fresh_results if r['method'] == method]
        if not rows:
            continue
        n = len(rows)
        successes = sum(r['independent_success'] is True for r in rows)
        all_hits = sum(r['all_training_hit'] for r in rows)
        tail_hits = sum(r['from_00145_hit'] for r in rows)
        new_methods.append({'method': method, 'n_trials': n,
                            'successes': successes, 'success_rate_pct': 100*successes/n,
                            'n_pose_valid': sum(r['pose_valid'] for r in rows),
                            'all_training_hits': all_hits, 'all_training_hit_rate_pct': 100*all_hits/n,
                            'all_training_wilson95_pct': wilson(all_hits, n),
                            'from_00145_hits': tail_hits, 'from_00145_hit_rate_pct': 100*tail_hits/n,
                            'from_00145_wilson95_pct': wilson(tail_hits, n)})
    comparison = {'previous_root': str(previous), 'new_trials': fresh_results,
                  'new_methods': new_methods, 'cumulative_formal_trials': n_trials,
                  'cumulative_started_trials': n_started,
                  'unchanged_training_calibration': all_s['calibration'] == old_stats['all_training']['calibration']}
    (ROOT/'results/cohort_extension.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n')
    with (ROOT/'results/new_trials.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fresh_results[0]))
        writer.writeheader(); writer.writerows(fresh_results)

    lines = [
        '# 起始点热力图与训练匹配：'+('纳入已确认的出画 C 点' if off_frame else '加入 2026-09-10 正式结果'), '',
        f'生成时间：{made}（北京时间）。加入用户确认的 {len(new_trials)} 次正式试验：5 次成功、1 次失败；当晚序号 1、2、3、7、10 共 5 次标为 smoke，继续排除。第 6 次 `2456b817d0a3-g1` 已使用更正后的 `visible_04`。', '',
        f'累计 **{n_trials} 次正式试验、{n_started} 次启动 VLA、{n_pose} 次可估计首帧位姿**。训练参考沿用 238 段首帧和包含 episode 145 的 93 段子集；有效训练位姿仍为 {all_s["reference_pose_valid"]}/238 与 {tail_s["reference_pose_valid"]}/93。原 250 张图的角点、位姿和原试验匹配结果保持不变，旧报告保留。', '',
        '**累计匹配统计**', '',
        '| 方法 | 可估计 / 全部正式试验 | 全集覆盖率 ↑ | ≥00145 覆盖率 ↑ | 全集平均 D ↓ | ≥00145 平均 D ↓ |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    if off_frame:
        sid = off_frame['selected_sample_ids'][0]
        p = next(p for p in poses if p['sample_id']==sid)
        c = next(q for q in p['annotation']['points'] if q['label']=='C')
        a, b = (by_reference[name][p['trial_id']] for name in ['all_training','from_00145'])
        note = [
            '**本轮确认的出画补估**', '',
            f"Full VLN / visible_04 的 C=({c['u']},{c['v']})、主观半径 {c['uncertainty_px']} px 已按用户确认原样纳入。A/B/D、原始 640×480 图像和内参保持原值，空白扩展画布仅用于表示原像素坐标。采用独立的 `off_frame_estimates.jsonl` 覆盖层；其余 255 张图和其余 18 次试验的匹配值均不变。", '',
            f"该点由 Astra low 子代理按首帧可见边界趋势提出，再经用户看图确认；这是一次出画推断，不是直接可见测量，也没有针对该 C 点的第二次独立估计。此前六图的独立复标按不外推规则将 C 留空。图中以 **04*†、空心菱形**标记，并保持非严格候选。", '',
            f"重投影 RMS={number(p.get('reprojection_rms_px'))} px；相机相对前壁 X={number(p.get('state',[None]*3)[0])} m、法向距离 Z={number(p.get('state',[None]*3)[1])} m、光轴朝向={number(p.get('state',[None]*3)[2])}°。最近距离 D：全集 {number(a['normalized_distance'])}、≥00145 {number(b['normalized_distance'])}；共用阈值 τ={all_s['calibration']['threshold']:.6f}。", '',
            f"[已确认的角点预览]({previous}/previews/off_frame_img_0253/annotation_preview.png) · [本次候选记录](annotations/off_frame_estimates.jsonl) · [用户确认与来源](provenance/off_frame_confirmation.json)", '',
        ]
        lines[6:6] = note
    for method in METHODS:
        a, b = methods['all_training'][method], methods['from_00145'][method]
        lines.append(f"| {NAMES[method]} | {a['n_pose_valid']}/{a['n_trials']} | {fraction(a['n_neighborhood_hits'],a['n_trials'])} | {fraction(b['n_neighborhood_hits'],b['n_trials'])} | {number(a['mean_normalized_distance'])} | {number(b['mean_normalized_distance'])} |")
    lines += ['', '覆盖率是“已启动、位姿可估计且最近训练距离 D≤阈值”的数量除以全部正式试验；D 的均值仅统计可估计位姿。缺失位姿保留在覆盖率分母，不能据此判定真实位置不匹配。匹配覆盖率与操作成功率是两个指标。', '',
              '![全训练集起始点热力图](figures/all_training/position_density.png)', '',
              '![episode≥145 起始点热力图](figures/from_00145/position_density.png)', '',
              '数字是 visible 条件编号，**04*、05* 表示本轮新增点**。颜色表示方法，实心圆为严格有效位姿，空心圆为其他基础有效候选，空心菱形表示存在推断角点；箭头为胸相机光轴在 XZ 平面的投影。背景为训练 KDE，所有面板与两套参考使用共同坐标范围、核协方差和色标；不为少量实验点拟合密度。没有有效位姿的试验不画虚构点，但计入面板分母。', '',
              '**本次新增结果**', '',
              '| 方法 | 操作成功率 | 可估计位姿 | 全集覆盖率 | ≥00145 覆盖率 |',
              '|---|---:|---:|---:|---:|']
    for m in new_methods:
        lines.append(f"| {NAMES[m['method']]} | {fraction(m['successes'],m['n_trials'])} | {m['n_pose_valid']}/{m['n_trials']} | {fraction(m['all_training_hits'],m['n_trials'])} | {fraction(m['from_00145_hits'],m['n_trials'])} |")
    lines += ['', '| 9/10 开始时间 | 方法 / 条件 | 试验 ID | 人工结果 | 位姿 | 全集最近 D / 命中 | ≥00145 最近 D / 命中 |',
              '|---|---|---|---|---|---|---|']
    for row in fresh_results:
        def match(prefix):
            distance = row[f'{prefix}_distance']
            return '缺失' if distance is None else f"{distance:.3f} / {'是' if row[f'{prefix}_hit'] else '否'}"
        pose = by_pose[row['trial_id']]
        quality = '严格有效' if row['strict_pose_valid'] else '候选有效' if row['pose_valid'] else '缺失：'+row['failure_reason']
        lines.append(f"| {row['start_time'][11:19]} | {NAMES[row['method']]} / {row['case_id']} | `{row['trial_id']}` | {'成功' if row['independent_success'] else '失败'} | {quality} | {match('all_training')} | {match('from_00145')} |")
    lines += ['', '**与上一版对比（相同参考和阈值）**', '',
              '| 方法 | 全集覆盖率：上一版 → 当前 | ≥00145 覆盖率：上一版 → 当前 |', '|---|---:|---:|']
    for method in METHODS:
        values = []
        for name in ['all_training', 'from_00145']:
            old = next(m for m in old_stats[name]['methods'] if m['method'] == method)
            new = methods[name][method]
            values.append(f"{fraction(old['n_neighborhood_hits'],old['n_trials'])} → {fraction(new['n_neighborhood_hits'],new['n_trials'])}")
        lines.append(f"| {NAMES[method]} | {values[0]} | {values[1]} |")
    lines += ['', '**标注与质量复核**', '',
              f'6 张新增首帧随机匿名化，由两位 **gpt-6-astra / low 子代理**分批逐图标注，再由第三位同配置子代理对全部 6 张独立复标；方法、成败和匹配分数均不提供给标注者。可比的 {len(differences)} 个角点，两轮差值中位数 {np.median(differences):.2f} px、最大 {max(differences):.2f} px。{len(review["conflict_sample_ids"])} 张触发 >6 px 或可见性分歧，由另一位 Astra low 看原图裁定，历史两份记录保留。模型复标一致性不等于物理真值精度。', '',
              '沿用篮子长边前壁本体四角，上宽 26 cm、下宽 23 cm、侧长 8.5 cm。药瓶遮挡可基于首帧可见边界提出 inferred 候选并声明像素半径；出画不外推，不借用后续帧。', '',
              '| 新样本 | 方法 / 条件 | 角点或几何问题 | RMS (px) | XZ 扰动 p95 (cm) | 朝向扰动 p95 (°) |',
              '|---|---|---|---:|---:|---:|']
    for p in new_poses:
        issue = p['failure_reason'] or ('推断角点 '+','.join(p['inferred_corners']) if p['inferred_corners'] else '四角可见')
        lines.append(f"| {p['sample_id']} | {NAMES[p['method']]} / {p['case_id']} | {issue} | {number(p.get('reprojection_rms_px'),2)} | {number(100*p['perturbation_xz_p95_radius_m'] if p.get('perturbation_xz_p95_radius_m') is not None else None,1)} | {number(p.get('perturbation_heading_p95_deviation_deg'),1)} |")
    lines += ['', f'当前严格有效实验 {n_strict}/{n_started}。基础位姿需完整四角、凸四边形面积≥80 px²、最短边≥6 px、正相机深度和前壁外侧、重投影 RMS≤3 px。严格筛选另要求四角可见、无检出的多解歧义，128 次像素圆盘扰动中求解率≥80%、XZ p95≤15 cm、朝向 p95≤15°。这些扰动不包含尺寸、标定或形状模型误差。', '',
              '![覆盖率与严格敏感性](figures/coverage_comparison.png)', '',
              '| 方法 | 全集严格覆盖率 | ≥00145 严格覆盖率 | 全集 MMD²ᵤ | ≥00145 MMD²ᵤ |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        a, b = methods['all_training'][method], methods['from_00145'][method]
        lines.append(f"| {NAMES[method]} | {fraction(a['strict_neighborhood_hits'],a['n_trials'])} | {fraction(b['strict_neighborhood_hits'],b['n_trials'])} | {number(a['mmd2_unbiased'])} | {number(b['mmd2_unbiased'])} |")
    calibration = all_s['calibration']
    lines += ['', f'匹配状态为胸相机相对篮子前壁的 [X,Z,光轴朝向]，朝向差环绕至 ±180°。与上一版完全相同：sX={calibration["scales"][0]:.6f} m、sZ={calibration["scales"][1]:.6f} m、sψ={calibration["scales"][2]:.6f}°，τ={calibration["threshold"]:.6f}；新增实验不参与尺度、阈值、MMD 核宽或训练 KDE 协方差拟合。MMD²ᵤ 为无偏估计，可为负，不是百分制概率。', '',
              f'保留原有训练几何诊断：全集 {diagnostic["diagnostic"]["all_training"]["n_negative_wall_y"]} 个候选的前壁 Y 为负，图中红叉标注；实验 {diagnostic["diagnostic"]["experiments"]["n_negative_wall_y"]} 个。此分析衡量相机与对象的几何邻近，不能解释为底盘位姿、完整观测一致或操作成功保证。方法的样本量小且不等；ObjectNav 输入与 VLN 不同，单独解释。这批 T1 全流程数据仍不直接填入表 2 的配对近场消融。', '',
              '**文件与复算**', '',
              f'- [上一版报告]({previous}/report.md)。',
              '- [复算说明](README.md)；[逐试验变更与新增统计](results/cohort_extension.json)；[新增六次 CSV](results/new_trials.csv)。',
              '- [全集方法 CSV](results/all_training/methods.csv)；[全集全部试验](results/all_training/trials.csv)；[全集 JSON](results/all_training/statistics.json)。',
              '- [≥00145 方法 CSV](results/from_00145/methods.csv)；[≥00145 全部试验](results/from_00145/trials.csv)；[≥00145 JSON](results/from_00145/statistics.json)。',
              f'- [{len(poses)} 张标注复核页](annotation_gallery.html)；[新增复标比较](provenance/new_annotation_comparison.json)；[位姿 CSV](results/poses.csv)；[像素敏感性图](figures/pixel_sensitivity.png)。',
              '- [几何与规则](analysis_config.json)；[纳入及排除清单](preflight.json)；[最终完整性验证](provenance/final_validation.json)。']
    for method in METHODS:
        lines.append(f"- {NAMES[method]} 最近训练图：[全集](figures/all_training/nearest_{method}.png)，[≥00145](figures/from_00145/nearest_{method}.png)。")
    document='\n'.join(lines)+'\n'
    if off_frame:
        document=document.replace('出画不外推，不借用后续帧。', '此次仅对用户确认的 img_0253/C 启用出画外推，其他未确认的越界角点仍拒绝；不借用后续帧。')
    (ROOT/'report.md').write_text(document)
    print(json.dumps({'report': str(ROOT/'report.md'), 'new_trials': len(new_trials),
                      'formal_trials': n_trials, 'estimable_experiments': n_pose}))
