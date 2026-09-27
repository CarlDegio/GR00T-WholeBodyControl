"""Report explicitly inferred bottle-occluded corners alongside the prior run."""
from pathlib import Path
import json
import math

from paths import ROOT
from metrics import METHODS

NAMES = {'full_vln':'Full VLN','nav_direct_vla':'Nav → VLA',
         'geometric_vla':'Geometric → VLA','full_objectnav':'Full ObjectNav'}
REFERENCES = [('all_training','全训练集'),('from_00145','episode ≥ 00145')]


def rate(row, strict=False):
    hits = row['strict_neighborhood_hits' if strict else 'n_neighborhood_hits']
    return f"{hits}/{row['n_trials']}（{100*hits/row['n_trials']:.1f}%）"


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def generate(config, poses, summaries, diagnostic, made):
    revision=config['occlusion_revision']
    previous=Path(revision['previous_revision'])
    before={p['sample_id']:p for p in json.loads((previous/'results/poses.json').read_text())}
    additions=[p for p in poses if p.get('occlusion_estimated')]
    experiments=sorted([p for p in additions if p['split']=='experiment'],key=lambda p:(METHODS.index(p['method']),p['case_id']))
    all_s,tail_s=summaries['all_training'],summaries['from_00145']
    calibration=all_s['calibration']
    n_exp=sum(p['split']=='experiment' and p['valid'] for p in poses)
    n_strict=sum(p['split']=='experiment' and p['strict_valid'] for p in poses)
    n_points=sum(len(p['occlusion_estimated_labels']) for p in additions)
    rows=[]
    for p in additions:
        rows.append({'sample_id':p['sample_id'],'split':p['split'],'episode_id':p['episode_id'],
                     'method':p['method'],'case_id':p['case_id'],
                     'previous_valid':before[p['sample_id']]['valid'],'current_valid':p['valid'],
                     'failure_reason':p['failure_reason'],'estimated_labels':p['occlusion_estimated_labels'],
                     'points':p['annotation']['points'],'strict_valid':p['strict_valid'],
                     'reprojection_rms_px':p.get('reprojection_rms_px'),
                     'negative_wall_y':p.get('camera_position_object_m',[0,0,0])[1]<0,
                     'perturbation_xz_p95_radius_m':p.get('perturbation_xz_p95_radius_m'),
                     'perturbation_heading_p95_deviation_deg':p.get('perturbation_heading_p95_deviation_deg')})
    repeat=[]
    by_id={p['sample_id']:p for p in poses}
    for path in sorted((ROOT/'annotations').glob('occlusion_repeat_*.jsonl')):
        for line_no,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            assert row['annotator']=={'model':'gpt-6-astra','reasoning_effort':'low'}
            main=by_id[row['sample_id']]
            for point in row['points']:
                if point['label'] not in main['occlusion_estimated_labels']:continue
                primary=next(q for q in main['annotation']['points'] if q['label']==point['label'])
                delta=None if point['u'] is None else math.hypot(point['u']-primary['u'],point['v']-primary['v'])
                repeat.append({'sample_id':row['sample_id'],'label':point['label'],
                               'displacement_px':delta,'primary_radius_px':primary['uncertainty_px'],
                               'repeat_radius_px':point.get('uncertainty_px'),
                               'within_primary_radius':delta is not None and delta<=primary['uncertainty_px'],
                               'repeat_source':str(path),'line':line_no})
    comparison={}
    for name,_ in REFERENCES:
        old=json.loads((previous/f'results/{name}/statistics.json').read_text())
        comparison[name]={'previous_reference_valid':old['reference_pose_valid'],
                          'current_reference_valid':summaries[name]['reference_pose_valid'],
                          'previous_calibration':old['calibration'],'current_calibration':calibration,
                          'methods':[{'method':a['method'],'previous':a,'current':b}
                                     for a,b in zip(old['methods'],summaries[name]['methods'])]}
    audit={'previous_revision':str(previous),'counts':{'selected':revision['selected_counts'],
           'estimated_samples':len(additions),'estimated_points':n_points,
           'new_valid_train':sum(p['split']=='train' and p['valid'] for p in additions),
           'new_valid_experiments':sum(p['split']=='experiment' and p['valid'] for p in additions)},
           'samples':rows,'independent_repeat_points':repeat,
           'comparison':comparison,
           'scope':'Exploratory shape-prior estimates. Repeat agreement does not validate physical accuracy. Calibration is refit on the expanded training reference using the unchanged rule.'}
    (ROOT/'results/occlusion_comparison.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    lines=[
        '# 胸视角训练起始分布匹配：加入药瓶遮挡角点估计','',
        f'生成时间：{made}。按用户要求，由 **gpt-6-astra / low 子代理**对 {len(additions)} 张首帧中的 {n_points} 个药瓶遮挡角点补估。沿用左图篮子本体四角和 **26 / 23 / 8.5 cm** 尺寸。原标注文件保留；已有非空坐标、可见性和不确定半径均未改变，补估另存为独立覆盖层。','',
        f'同一规则应用于 4 张实验图和 17 张训练图。新增有效位姿为实验 {audit["counts"]["new_valid_experiments"]}/4、训练 {audit["counts"]["new_valid_train"]}/17；现在实验首帧可估计 **{n_exp}/12**，全训练集 **{all_s["reference_pose_valid"]}/238**，00145 起子集 **{tail_s["reference_pose_valid"]}/93**。通过基础几何检查的补估点已纳入热图和匹配计算，仍明确标记为 inferred。','',
        '![实验遮挡角点补估](figures/occlusion_estimate_review.png)','',
        '上图仅裁取原始首帧局部以便核对，坐标仍以完整 640×480 原图为准；空心角点和虚线表示推断，圆圈为标注者给出的主观像素范围。它不是可见角点的直接测量，也不是统计置信区间。','',
        '| 方法 | 可估计 / 全部正式试验 | 全集覆盖率 ↑ | ≥00145 覆盖率 ↑ | 全集平均最近距离 D ↓ | ≥00145 平均 D ↓ |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for a,b in zip(all_s['methods'],tail_s['methods']):
        lines.append(f"| {NAMES[a['method']]} | {a['n_pose_valid']}/{a['n_trials']} | {rate(a)} | {rate(b)} | {fmt(a['mean_normalized_distance'])} | {fmt(b['mean_normalized_distance'])} |")
    lines += ['',
        '覆盖率分子为已启动、可估计且进入训练邻域的试验数，分母保留全部 13 次正式试验，包括 Geometric 的一次未启动；smoke 排除。连续距离只在可估计样本上计算。每种方法只有 3 次已启动试验，本批次结果不能证明总体最优，也不是 experiment.md 表 2 的配对近场相机消融。','',
        '![全训练集空间热图](figures/all_training/position_density.png)','',
        '![00145起空间热图](figures/from_00145/position_density.png)','',
        '热图为训练名义位姿的 KDE；灰色点表示训练位置，灰色空心菱形是本轮新补估的训练位置，红叉标出上下关系异常的训练候选。实验按方法着色，**空心菱形是本轮遮挡补估**，空心圆是此前的非严格候选，实心圆通过严格检查。箭头为相机光轴的 XZ 投影方向。所有方法和两套参考共用坐标范围、核协方差和色标；没有给仅有 3 个实验点拟合平滑密度。','',
        '**与补估前对照。** 以下只改变遮挡点处理，尺寸、已有角点、原始首帧和正式试验清单相同。由于训练参考也扩充，按原规则重新拟合尺度和阈值；新旧距离 D 的数值尺度不同，不能把差值解释为物理精度提高。','',
        '| 方法 | 全集覆盖率：原版 → 补估版 | ≥00145 覆盖率：原版 → 补估版 |','|---|---:|---:|',
    ]
    for a,b in zip(comparison['all_training']['methods'],comparison['from_00145']['methods']):
        lines.append(f"| {NAMES[a['method']]} | {rate(a['previous'])} → {rate(a['current'])} | {rate(b['previous'])} → {rate(b['current'])} |")
    lines += ['',f'[补估前报告]({previous}/report.md)和原始产物保持不变；[逐图变更与新旧统计 JSON](results/occlusion_comparison.json)记录每个新增点和位姿质量。','',
        '**补估依据与验证范围。** Astra low 只看指定首帧以及原先的非空点，利用可见底边、上边、部分侧边和等腰梯形形状先验推断遮挡位置；标注任务不提供方法、成败、匹配分数或参考子集标签。不借用后续帧，不用 PnP 或实验排名反调角点。出画、目标面不可辨认等其他缺失维持原记录。','',
        '| 实验 | 新补点（原图像素） | 重投影 RMS | XZ 扰动 p95 | 朝向扰动 p95 |','|---|---|---:|---:|---:|',
    ]
    for p in experiments:
        points='；'.join(f"{q['label']}=({q['u']},{q['v']}) ±{q['uncertainty_px']}px" for q in p['annotation']['points'] if q['label'] in p['occlusion_estimated_labels'])
        lines.append(f"| {NAMES[p['method']]} {p['case_id']} | {points} | {fmt(p.get('reprojection_rms_px'),2)} px | {fmt(100*p.get('perturbation_xz_p95_radius_m',0),1)} cm | {fmt(p.get('perturbation_heading_p95_deviation_deg'),1)}° |")
    if repeat:
        distances=[p['displacement_px'] for p in repeat if p['displacement_px'] is not None]
        lines += ['',f'另一位 Astra low 对 {len(set(p["sample_id"] for p in repeat))} 张图独立补估复核，涵盖全部 4 张新增实验图以及 4 张训练图；未向其提供首轮补估坐标或几何结果。可比较的 {len(distances)} 个点最大分歧为 {max(distances):.2f} px，{sum(p["within_primary_radius"] for p in repeat)}/{len(repeat)} 在首轮主观像素半径内。复核记录单独保留，不据此移动主估计以改善匹配。它只说明同类模型的一致性，不是物理真值验证。','']
    lines += [
        '基础检查沿用四点完整且在图内、凸四边形面积≥80 px²、最短边≥6 px、正相机深度和相机位于前壁外侧、重投影 RMS≤3 px。每图按主观角点半径做 128 次独立均匀圆盘扰动，并保留名义解附近分支。小 RMS 不能证明隐藏角点正确；扰动范围也不覆盖尺寸、内参和形状模型误差，四点独立扰动未建模形状先验可能造成的相关误差。','',
        f'严格敏感性仍要求四点全可见、无检出的多解歧义且扰动稳定；本轮补估不会因此变为严格样本。严格实验仍为 **{n_strict}/12**；训练严格参考为全集 {all_s["strict_reference_pose_valid"]}、子集 {tail_s["strict_reference_pose_valid"]}。','',
        '下表严格覆盖率使用严格集合；MMD 使用含补估的基础集合，并非严格集合的 MMD。','',
        '| 方法 | 全集严格覆盖率 | ≥00145 严格覆盖率 | 全集补估版 MMD²ᵤ ↓ | ≥00145 补估版 MMD²ᵤ ↓ |','|---|---:|---:|---:|---:|',
    ]
    for a,b in zip(all_s['methods'],tail_s['methods']):
        lines.append(f"| {NAMES[a['method']]} | {rate(a,True)} | {rate(b,True)} | {fmt(a['mmd2_unbiased'])} | {fmt(b['mmd2_unbiased'])} |")
    lines += ['', '![覆盖率和严格敏感性](figures/coverage_comparison.png)','', '![像素扰动敏感性](figures/pixel_sensitivity.png)','',
        '**保留的质量问题。**','',
        '| 样本 | 处理与限制 |','|---|---|',
    ]
    for p in additions:
        label=f"训练 episode {p['episode_id']:06d}" if p['split']=='train' else f"{NAMES[p['method']]} {p['case_id']}"
        if not p['valid']:
            lines.append(f"| {label} / {p['sample_id']} | {p['failure_reason']}，RMS {fmt(p.get('reprojection_rms_px'))} px；保留标注，未纳入热图或匹配参考。 |")
        elif p['camera_position_object_m'][1]<0:
            lines.append(f"| {label} / {p['sample_id']} | 名义解前壁 Y={p['camera_position_object_m'][1]:.3f} m，上下关系异常；XZ 扰动 p95={p['perturbation_xz_p95_radius_m']:.3f} m。按未改变的基础规则保留为带警示的探索性候选，不进入严格结果。 |")
    lines += ['',
        '图上保留几何诊断标记，避免将含异常候选的全集结果误认作已实测验证的空间分布。没有强制选择 Y>0 的解或为更好排名放宽 RMS 阈值；00145 起子集和实验的 Y 符号检查详见诊断 JSON。','',
        '**坐标与指标。** 状态仍为胸相机相对于篮子前壁的 [X,Z,光轴朝向]。前壁高 H=√(0.085²−0.015²)=0.083666 m；四个物体点依次为 (−0.13,H,0)、(0.13,H,0)、(0.115,0,0)、(−0.115,0,0)，通过本地胸相机内参和 IPPE 平面 PnP 求解。Y 和俯仰另存，不进入距离；这不是机器人底盘或世界位姿，也不代表完整图像相似度。无需深度或训练新模型。','',
        r'\[D(q,T)=\min_{t\in T}\sqrt{((X_q-X_t)/s_X)^2+((Z_q-Z_t)/s_Z)^2+(\operatorname{wrap}_{180}(\psi_q-\psi_t)/s_\psi)^2}.\]','',
        f'两套参考共用 sX={calibration["scales"][0]:.6f} m、sZ={calibration["scales"][1]:.6f} m、sψ={calibration["scales"][2]:.6f}° 和 τ={calibration["threshold"]:.6f}，D≤τ 记为邻域命中。尺度规则为全部有效训练位姿的 1.4826×MAD 加原固定下限，τ 为留一最近邻距离的 95 分位；不根据实验结果选阈值。MMD²ᵤ 仍是共享带宽的无偏高斯核估计，可为负且不是百分制概率；阈值敏感性和 Wilson 区间保存在统计 JSON。','',
        '**产物与复算。** [README](README.md)给出完整执行命令和数据来源。所有原视频、训练首帧和实际 VLA 首次输入胸图保持不变，几何测量和内参历史假设沿用[原方法报告]('+str(previous)+'/report.md)。','',
        '- [250 张标注复核页](annotation_gallery.html)；[补估与复核对照](results/occlusion_comparison.json)。',
        '- [逐图位姿 CSV](results/poses.csv)；[完整候选及扰动 JSON](results/poses.json)。',
        '- [全集方法 CSV](results/all_training/methods.csv)；[全集逐试验 CSV](results/all_training/trials.csv)；[全集统计](results/all_training/statistics.json)。',
        '- [子集方法 CSV](results/from_00145/methods.csv)；[子集逐试验 CSV](results/from_00145/trials.csv)；[子集统计](results/from_00145/statistics.json)。',
        '- [几何诊断](results/geometry_diagnostics.json)；[分析设置](analysis_config.json)；[变更范围](provenance/occlusion_revision.json)。',
    ]
    for method in METHODS:
        lines.append(f"- {NAMES[method]} 原图配对：[全集](figures/all_training/nearest_{method}.png)，[≥00145](figures/from_00145/nearest_{method}.png)。")
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(ROOT/'report.md'),'added_estimates':audit['counts'],'independent_repeat_points':len(repeat)}))
