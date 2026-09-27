# GATE 论文 teaser：三版设计稿

目标论文：`/home/user/Project/Overleaf/6a925da904325a8d60afac0b`，当前题目 *Beyond Arrival: Geometric Alignment for Agentic Humanoid Loco-manipulation*。

设计目标：从真实 G1 任务场景出发，连贯说明“导航到达后的视野/位姿未必适合操作 → GATE 修正交接状态 → 完整任务成功率提高”。本目录是设计比较稿，未修改论文正文和实验记录。

## 三个方向

- **A_scene_and_evidence（推荐）**：主场景 + 放大的头部相机前后对照 + 完整任务成功率。视觉层次最直接，优先适合首页。
- **B_mechanism_and_policy**：主场景 + T1 前后实拍 + 几何反馈俯视示意 + 同操作策略近场结果。更清楚地区分模块效果与系统效果。
- **C_repeated_handoffs**：主场景居中，拿袋与桶前对齐两侧展开，突出长任务中的多次交接。左侧为论文已有 pickup example，不声称与中央主图是同一次同步录像。

每版有 `*_preview.jpg`、高清 PNG、PDF 和 SVG。PDF 页宽 177.8 mm、高约 72.7 mm，与正文 70 mm teaser 预留接近；严格放入 70 mm 高框时应等比例缩至约 171.1 mm 宽。标题和说明仍可在选版后进一步精简。SVG 的文字、箭头、图表可编辑，照片为内嵌位图。源脚本 `build_concepts.py`。

## 真实素材

主图取自 V6 无小窗版，与 V8 使用相同场景、透明机器人和虚线路径；V8 已核验的头部图移到侧边放大。全部照片直接置入版面，没有生成式重画。轨迹和机器人合成位置是视觉示意，不是测量轨迹。

- `../v6_trajectories/teaser_v6_clean.png`
- `../v8_front_rim/head_5_annotated.png`
- `../v8_front_rim/head_6_annotated.png`
- 论文 `figures/gate_before_after/t1_{before,after}.png`
- 论文 `figures/gate_before_after/t4_bag_{before,after}.png`

V8 绿色目标蒙版和前横沿线来自原提示词 `green rubbish bin` 的离线 YOLOE/Hough 检测，不能称为当时在线控制器的原始叠加。前后照片说明操作前的交接观测，不直接证明后续物理成功。完整来源与 SHA-256 见 `sources_and_claims.json`。

## 数值依据

当前论文 `supplementary.tex` 的 `tab:main`（约 730 行）、`tab:g1_method_configuration`（约 767 行）及 `tab:near_field_results`（约 981 行）：

| 指标 | Direct handoff / Direct VLA | Full agent + GATE / GATE + VLA |
|---|---:|---:|
| 完整任务 T1–T4 合计 | 6/40 = 15.0% | 33/40 = 82.5% |
| 完整任务 T1 | 6/10 = 60% | 9/10 = 90% |
| 完整任务 T2 | 0/10 = 0% | 9/10 = 90% |
| 完整任务 T3 | 0/10 = 0% | 8/10 = 80% |
| 完整任务 T4 | 0/10 = 0% | 7/10 = 70% |
| 近场 T1，同一操作策略 | 24/50 = 48.0% | 45/54 = 83.3% |
| 近场 T2，同一操作策略 | 8/43 = 18.6% | 49/53 = 92.5% |

总体差值是 **67.5 个百分点**。Direct VLA 同时绕过 GATE 与相应交接检查，不能把这组系统比较解释为只消融了几何控制器。近场数据属于不同试验队列，起始状态不是逐一匹配。此次没有重新汇总归档副本，也没有修改标注或统计。

## 可继续加的内容（择一，避免拥挤）

1. **观测分布证据**：T1/T2 对齐后到操作参考观测的视觉距离，其逐试验变化中位数分别 −36.1% / −35.2%；60/62 与 59/60 对下降。来源是论文 `figures/data/alignment_visual_change.json` 和当前正文，不能使用旧 `RESULTS_SOURCES.md` 中过时的 T2 30 对统计。可以做一个很小的 Before → After 距离图，以支持“为什么有效”。该指标并非几何位姿误差。
2. **真实失败交接与成功交接**：可使用实际 VLA 首帧中“目标越出视野”与“目标完整居中”的对照，并标注 representative trials。两个试验不应伪装成同一次 before/after。
3. **跨任务操作缩略图**：搬箱、端碗/倒碗与踩桶展示全身技能范围；它们可以说明任务多样性，不能单凭照片声称未见场景泛化。

推荐终稿以 **A 的整体结构 + B 的小几何图** 合并，保持一张主图、一个真实前后对照、一个机制小图和一项系统结果。OVMM 可留给正文，不必让首页同时承担另一平台的复杂实验条件。

## 可用图注草稿（英文）

*GATE bridges navigation and whole-body manipulation through task-conditioned geometric alignment. A real G1 task montage illustrates bag pickup, transport, and pedal-assisted disposal; overlaid poses and paths are schematic. Egocentric observations show how alignment changes the manipulation handoff. Across four real-world tasks, the full agent achieves 82.5% success (33/40), compared with 15.0% (6/40) for direct handoff, which bypasses alignment and the associated readiness checks. The same navigation and task manipulation policies are used in this comparison. Detection overlays in the bin example are generated offline from recorded camera frames.*

B 追加：*Separate near-field cohorts compare the same task policy with and without preceding alignment; initial configurations are not paired across methods.*

C 追加：*The side panels illustrate task stages and are not asserted to form a synchronized sequence with the central montage.*
