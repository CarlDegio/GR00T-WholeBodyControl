"""Generate the two-reference report from the current geometry and statistics."""
import collections
import datetime
import importlib.metadata
import json
import math
import platform

import cv2
import numpy as np

from paths import ROOT
from metrics import METHODS

NAMES = {'full_vln': 'Full VLN', 'nav_direct_vla': 'Nav → VLA',
         'geometric_vla': 'Geometric → VLA', 'full_objectnav': 'Full ObjectNav'}
REFERENCES = [('all_training', '全训练集'), ('from_00145', 'episode ≥ 00145')]


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def coverage(row, strict=False):
    key = 'strict_neighborhood_hits' if strict else 'n_neighborhood_hits'
    rate = 'strict_hit_rate_pct' if strict else 'neighborhood_hit_rate_pct'
    return f"{row[key]}/{row['n_trials']}（{row[rate]:.1f}%）"


def main():
    config = json.loads((ROOT / 'analysis_config.json').read_text())
    geometry = config['geometry_m']
    height = math.sqrt(geometry['side']**2 - ((geometry['top_width']-geometry['bottom_width'])/2)**2)
    poses = json.loads((ROOT / 'results/poses.json').read_text())
    summaries = {name: json.loads((ROOT / 'results' / name / 'statistics.json').read_text()) for name, _ in REFERENCES}
    all_s, tail_s = summaries['all_training'], summaries['from_00145']
    methods = {name: {m['method']: m for m in s['methods']} for name, s in summaries.items()}
    calibration = all_s['calibration']
    diagnostic = json.loads((ROOT / 'results/geometry_diagnostics.json').read_text())
    made = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    environment = {'generated_at': made, 'python': platform.python_version(), 'opencv': cv2.__version__,
                   'packages': {p: importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','pillow','av']}}
    (ROOT / 'provenance/environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    if config.get('cohort_extension'):
        from report_extension import generate
        generate(config, poses, summaries, diagnostic, made)
        return
    repeat = json.loads((ROOT / 'provenance/repeatability_comparison.json').read_text())
    displacements = [d for row in repeat for d in row['point_displacements_px']]
    conflicts = [row for row in repeat if row['different_missing_points'] or (row['max_point_displacement_px'] or 0) > 6]
    if config.get('occlusion_revision'):
        from report_occlusion import generate
        generate(config,poses,summaries,diagnostic,made)
        return
    strict_exp = sum(p['split']=='experiment' and p['strict_valid'] for p in poses)
    lines = [
        '# 胸视角训练起始分布匹配：本体四角与修正尺寸', '',
        f'生成时间：{made}。使用用户最后确认的**左图本体四角**：上宽 {100*geometry["top_width"]:g} cm、下宽 {100*geometry["bottom_width"]:g} cm、两侧斜边 {100*geometry["side"]:g} cm。复用全部已完成的 Astra low 标注；标注文件逐份 SHA-256 与原版一致，没有根据新尺寸移动任何点，也没有使用中途的外沿重标结果。', '',
        f'238 段胸视频已完整下载并通过远端/本地 SHA-256 比对；训练一律取首个解码帧，正式实验取实际首次送入 VLA 策略的胸图。共 238 张训练首帧、12 张实验图，全部标注与复核均由 **gpt-6-astra / low 子代理**完成。匹配由单目几何和统计计算，不需要训练额外模型，也没有使用训练深度或 FAST-LIO。', '',
        f'在当前几何下，可求解训练样本 {all_s["reference_pose_valid"]}/{all_s["reference_source_frames"]}，子集 {tail_s["reference_pose_valid"]}/{tail_s["reference_source_frames"]}，实验 8/12 张。此前的相机上下关系异常已消除：有效训练和实验解的前壁 Y 坐标均为正；这来自尺寸修正后的最小重投影误差解，**没有把 Y>0 加入筛选条件**。这项一致性检查不等于独立的物理精度验证。', '',
        '| 方法 | 可估计位姿 / 全部正式试验 | 全训练集覆盖率 ↑ | ≥00145 覆盖率 ↑ | 全集平均最近距离 D ↓ | ≥00145 平均 D ↓ |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        a, b = methods['all_training'][method], methods['from_00145'][method]
        lines.append(f"| {NAMES[method]} | {a['n_pose_valid']}/{a['n_trials']} | {coverage(a)} | {coverage(b)} | {fmt(a['mean_normalized_distance'])} | {fmt(b['mean_normalized_distance'])} |")
    best_text = []
    for name, label in REFERENCES:
        rows = summaries[name]['methods']; maximum = max(row['neighborhood_hit_rate_pct'] for row in rows)
        names = '、'.join(NAMES[row['method']] for row in rows if row['neighborhood_hit_rate_pct']==maximum)
        best_text.append(f'{label}中可确认覆盖率最高的是 {names}（{maximum:.1f}%）')
    lines += [
        '', '；'.join(best_text)+'。这个排序仅描述本批次的观测，不能证明某种方法总体最优。Full VLN 的连续最近距离较小，但只有 1 张有效图；另外两张下角被药瓶遮挡。不能把缺失几何当作物理不匹配，也不能用单个有效样本代表完整分布。', '',
        '覆盖率分子为“已启动、具有可用位姿、且落入参考训练邻域”的试验数，分母保留全部正式试验。连续距离只在可估计样本上计算，必须连同有效样本数阅读。当前覆盖率同时受到位姿接近程度与可观测性的影响，不能理解为完整视觉相似度或操作成功概率。', '',
        '![两套覆盖率与严格筛选](figures/coverage_comparison.png)', '',
        '![全训练集密度与实验位置](figures/all_training/position_density.png)', '',
        '![00145起子集密度与实验位置](figures/from_00145/position_density.png)', '',
        '热图背景为训练首帧的二维核密度，灰点为训练位置，彩点为各方法的实验位置；箭头为胸相机光轴投影方向，编号对应 visible_01/02/03。实心点满足严格筛选，空心点只通过基础几何检查。两套参考和所有方法共用坐标范围、核协方差及色标。方法各只有 3 个已启动样本，因此以散点呈现；不为它们拟合平滑密度。匹配距离还包含朝向，不能仅从二维热图判断命中。', '',
        '**样本范围与缺失。** 正式试验清单沿用原冻结 preflight：共 13 次 T1 可见组全流程试验、12 次实际启动 VLA，每方法 3 张胸图。Geometric 的 `b3df1b125f7e-g1` 在 VLA 前超时，无策略输入胸图，仍保留在其 4 次分母；smoke 和已有排除项不计入。该批次不是 experiment.md 表 2 的配对近场相机消融，不能直接填入该表。ObjectNav 的导航输入与 VLN 不同，单列展示，不将差异单独归因于 BasePose。', '',
        '| 参考范围 | 原始首帧 | 可估计 | 严格筛选 | 角点无法定位 | 重投影误差超限 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, label in REFERENCES:
        rows = [p for p in poses if p['split']=='train' and (name=='all_training' or p['episode_id']>=145)]
        errors = collections.Counter(p['failure_reason'] for p in rows if not p['valid'])
        s = summaries[name]
        lines.append(f"| {label} | {len(rows)} | {s['reference_pose_valid']} | {s['strict_reference_pose_valid']} | {errors['unresolvable_front_wall_corners']} | {errors['high_reprojection_error']} |")
    lines += [
        '', '所有训练首帧均进入流程，但只有可求解子集进入空间热图。不能宣称已恢复全部 238 个真实起始位姿；丢失并非随机，也可能改变观测到的训练分布。未用后续帧替代出画或被遮挡的首帧。实验的 4 张几何缺失和另 1 次未启动分别保留，见逐试验 CSV。', '',
        '**物体模型与坐标。** 上角 A/B 取凸出外沿下方的前壁本体转折，下角 C/D 取同一长前壁本体底边两端；圆角用相邻直边延长线交点。根据给定的等腰梯形，上下边差的一半为 1.5 cm，侧边 8.5 cm 对应的面内高为：', '',
        rf'\[H=\sqrt{{({geometry["side"]})^2-\left(\frac{{{geometry["top_width"]}-{geometry["bottom_width"]}}}{{2}}\right)^2}}={height:.8f}\ \mathrm{{m}}.\]', '',
        rf'\[ A=(-{geometry["top_width"]/2:g},H,0),\ B=({geometry["top_width"]/2:g},H,0),\ C=({geometry["bottom_width"]/2:g},0,0),\ D=(-{geometry["bottom_width"]/2:g},0,0).\]', '',
        '原点为下边中点，X 沿 D→C，Y 沿前壁向上，Z 指向前壁外侧。由四个二维/三维对应点使用 OpenCV IPPE 求平面 PnP，并分别细化候选解，保存所有物理解；相机中心 C=-Rᵀt。状态 q=[Cₓ,C_z,ψ]，ψ 为相机光轴绕前壁 Y 的朝向，0° 表示朝向前壁。该量是**相机相对于篮子前壁**的几何；没有相机到底盘、篮子到地面的外参时，不能称为机器人底盘位姿或地面 yaw。[OpenCV PnP 文档](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)', '',
        '胸内参使用冻结的本地 640×480 标定：fx=607.461181640625、fy=607.4806518554688、cx=312.74774169921875、cy=253.5120391845703，保存的五项畸变系数为 0。按用户指定也应用于训练视频；远端元数据未独立确认训练时相机序列号和预处理。相同分辨率不证明历史内参完全相同。', '',
        '**数值定义。** 每个可估计实验状态到参考集的最近归一化距离为：', '',
        r'\[D(q,T)=\min_{t\in T}\sqrt{((x_q-x_t)/s_x)^2+((z_q-z_t)/s_z)^2+(\operatorname{wrap}_{180}(\psi_q-\psi_t)/s_\psi)^2}.\]', '',
        f'共享尺度为 sₓ={calibration["scales"][0]:.6f} m、s_z={calibration["scales"][1]:.6f} m、sψ={calibration["scales"][2]:.6f}°。尺度按全部有效训练状态的 1.4826×MAD 拟合，下限为 0.02 m、0.02 m、5°；朝向使用圆周中心与环绕差。阈值 τ={calibration["threshold"]:.6f} 是全部有效训练样本留一最近邻距离的 95 分位，D≤τ 为命中。', '',
        '拟合规则、阈值分位、几何质检和缺失处理沿用查看初版方法排名前固定的方案。尺寸更新后重算训练状态与由它们拟合的数值，不根据实验成败调整规则；两套参考共用本轮同一尺度和阈值。子集包含编号 145，即 episode_id≥145，共 93 段，并非按文件名字符串或任意一半划分。', '',
        '子集属于全集，因此同一实验的 D(全集)≤D(子集) 是必然集合关系，程序逐试验检查这一性质。两列数值差不能解释成“使用更多参考数据的算法更好”。相机 Y 和俯仰角保存在 JSON/CSV 中但不进入本指标；位置及朝向接近也不保证图像上下布局、药瓶位置、纹理、光照或完整 VLA 观测一致。', '',
        '**标注、质检与敏感性。** 全部原图逐张经 Astra low 查看。独立复标包括全部 12 张实验图和固定随机的 12 张训练图。', '',
        f'双方均能定位的 {len(displacements)} 个对应点，位移中位数 {np.median(displacements):.2f} px、95 分位 {np.quantile(displacements,.95):.2f} px；{len(conflicts)} 张存在 >6 px 或可定位性分歧，均由另一 Astra low 裁定。另做过 26 张长边投影较窄的图及 13 张几何质检图的复核，历史修订和逐行来源保留。它反映同类模型的一致性，不是人工真值准确率；本轮不重新标点。', '',
        '基础质检要求四点完整、在图内、形成凸四边形，面积≥80 px²、各边≥6 px、物体点在相机前方、相机在前壁外侧、重投影 RMS≤3 px。基础结果允许明确标记的推断角点。若另一个解的 RMS 与最优解相差≤0.5 px，且中心相差>10 cm 或朝向相差>15°，标记分支歧义。小重投影误差本身不能证明物理准确。', '',
        '每张可估计图使用标注者给出的 uncertainty_px 作为四点独立均匀圆盘扰动半径，做 128 次重解，保留名义解附近分支。严格筛选要求四点全可见、无检出的多解歧义、扰动求解率≥80%、XZ 位移的 95 分位≤15 cm、朝向偏移的 95 分位≤15°。这些范围是主观标注敏感性，不是测量置信区间；固定训练参考下的实验扰动命中比例也不覆盖参考位姿自身的不确定性。', '',
        '![像素扰动敏感性](figures/pixel_sensitivity.png)', '',
        '| 方法 | 严格位姿 / 全试验 | 全集严格覆盖率 | 子集严格覆盖率 | 全集条件命中率 | 子集条件命中率 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        a, b = methods['all_training'][method], methods['from_00145'][method]
        lines.append(f"| {NAMES[method]} | {a['n_strict_pose_valid']}/{a['n_trials']} | {coverage(a,True)} | {coverage(b,True)} | {fmt(a['conditional_hit_rate_pct'],1)}%（n={a['n_pose_valid']}） | {fmt(b['conditional_hit_rate_pct'],1)}%（n={b['n_pose_valid']}） |")
    lines += [
        '', f'严格参考同时缩减为全集 {all_s["strict_reference_pose_valid"]}、子集 {tail_s["strict_reference_pose_valid"]} 个训练样本，严格实验总共 {strict_exp} 张；仍共用本轮尺度与阈值。这改变了可用集合，用于说明结论对质检的敏感性，不能包装成完整数据的稳定排名。覆盖率误差棒为描述性的 Wilson 95% 区间，不含标定和标注误差；当前每方法 3 次启动的小样本也不足以支持显著性结论。', '',
        '| 方法 | 全集 τ×0.5 / τ / τ×2 覆盖率 | 子集 τ×0.5 / τ / τ×2 覆盖率 | 全集 MMD²ᵤ ↓ | 子集 MMD²ᵤ ↓ |',
        '|---|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        a, b = methods['all_training'][method], methods['from_00145'][method]
        values = lambda row: ' / '.join(f"{x['rate_pct']:.1f}%" for x in row['threshold_sensitivity'])
        lines.append(f"| {NAMES[method]} | {values(a)} | {values(b)} | {fmt(a['mmd2_unbiased'])} | {fmt(b['mmd2_unbiased'])} |")
    lines += [
        '', '补充 MMD²ᵤ 采用无偏双样本估计：在 [x/sₓ,z/s_z,cosψ/sψ(rad),sinψ/sψ(rad)] 上使用高斯核，带宽为全部有效训练嵌入的成对距离中位数。两套参考使用同一带宽；实验少于 2 个有效状态时不计算。有限样本无偏估计可以为负，不能变成百分制概率；本轮 Full VLN 因 n=1 留空。[Gretton 等的双样本检验论文](https://www.jmlr.org/papers/v13/gretton12a.html)', '',
        '**逐试验与原图证据。** [250 张标注复核页](annotation_gallery.html)可按实验/训练及严格/候选/缺失筛选，每张可打开原图和标注来源。原图配对按同一三维状态距离选取最近训练图，不做生成式补画、变形或视觉对齐。', '',
    ]
    for method in METHODS:
        lines.append(f"- {NAMES[method]}：[全训练集最近图](figures/all_training/nearest_{method}.png)，[≥00145 最近图](figures/from_00145/nearest_{method}.png)。")
    lines += ['', '[全训练集距离/朝向图](figures/all_training/heading_distance.png) · [子集距离/朝向图](figures/from_00145/heading_distance.png)。', '',
              '| 方法 / 条件 | 试验 ID | 位姿状态 | 全集最近 episode / D | 子集最近 episode / D |', '|---|---|---|---:|---:|']
    tails = {r['trial_id']: r for r in tail_s['trials']}
    for method in METHODS:
        for a in [r for r in all_s['trials'] if r['method']==method]:
            b = tails[a['trial_id']]
            status = '严格' if a['strict_pose_valid'] else ('候选' if a['pose_valid'] else ('未启动 VLA' if not a['vla_started'] else '角点无法定位'))
            nearest = lambda row: '—' if row['nearest_episode_id'] is None else f"{row['nearest_episode_id']:06d} / {row['normalized_distance']:.3f}"
            lines.append(f"| {NAMES[method]} / {a['case_id']} | {a['trial_id']} | {status} | {nearest(a)} | {nearest(b)} |")
    lines += [
        '', '**尺寸修正后的检查。** 给定左图本体尺寸与其角点现在对应。新求解仍只按重投影误差选择满足正深度和前侧约束的候选，没有加入 Y 正值约束。', '',
        '| 范围 | 旧尺寸下负 Y / 有效解 | 新尺寸下负 Y / 有效解 | 新 Y 中位数（m） |', '|---|---:|---:|---:|',
    ]
    for name, label in [('all_training','全训练集'),('from_00145','≥00145'),('experiments','实验')]:
        current = diagnostic['diagnostic'][name]; old = diagnostic.get('previous_geometry_diagnostic',{}).get(name)
        previous = '—' if old is None else f"{old['n_negative_wall_y']}/{old['n_solved']}"
        lines.append(f"| {label} | {previous} | {current['n_negative_wall_y']}/{current['n_solved']} | {fmt(current['median_wall_y_m'])} |")
    lines += [
        '', '该检查消除了此前系统性的上下关系疑点，不能代替相机—篮子的独立距离/角度测量。当前结果可作为单目已知尺寸几何的先导分析与图表，完整论文结论仍应结合预定且配对起点下更多重复试验、真实操作成败，以及少量已知摆位的尺度/方向验证。量尺与已知摆位即可完成后者，深度不是强制条件。不得为了排名而移动被遮挡角点、改变筛选阈值或忽略缺失。', '',
        '**产物与复现。** [README](README.md)给出完整执行命令；原始视频和首帧复用原目录，本修订目录保存独立结果和相同标注的副本，旧结果不覆盖。科学图提供 PNG/PDF/SVG。', '',
        '- [逐图位姿 CSV](results/poses.csv)；[全部候选、标注、扰动 JSON](results/poses.json)。',
        '- [全集方法 CSV](results/all_training/methods.csv)；[全集逐试验 CSV](results/all_training/trials.csv)；[全集统计 JSON](results/all_training/statistics.json)。',
        '- [子集方法 CSV](results/from_00145/methods.csv)；[子集逐试验 CSV](results/from_00145/trials.csv)；[子集统计 JSON](results/from_00145/statistics.json)。',
        '- [分析规则](analysis_config.json)；[尺度与阈值](results/calibration.json)；[KDE 设置](results/plot_settings.json)。',
        '- [用户尺寸确认](provenance/geometry_confirmation.json)；[原标注复用哈希](provenance/annotation_reuse.json)；[几何一致性统计](results/geometry_diagnostics.json)。',
        '- [数学检查](provenance/math_validation.json)；[软件环境](provenance/environment.json)；[代码快照](pipeline/)。', '',
    ]
    (ROOT / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(ROOT/'report.md'), 'geometry_m': geometry, 'reference_sizes': [all_s['reference_pose_valid'],tail_s['reference_pose_valid']], 'strict_experiments': strict_exp}))


if __name__ == '__main__':
    main()
