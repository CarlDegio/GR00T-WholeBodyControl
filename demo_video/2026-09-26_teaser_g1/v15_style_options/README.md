# GATE teaser V15：三种美化方案

## 2026-09-26：三个版本统一替换为具体方法对比

右下角原来的四项勾选清单已替换为 `Capability comparison`，比较原论文中的 **LaViRA、UniLM-Nav、Ours**。三个版本使用同一份数据，Ours 列单独着色。全部表头、行标题和勾叉仍是可编辑 PPT 对象。

| Capability | LaViRA | UniLM-Nav | Ours |
|---|:---:|:---:|:---:|
| Semantic navigation | ✓ | ✓ | ✓ |
| Training-free handoff | ✗ | ✓ | ✓ |
| Humanoid whole-body skills | ✗ | ✗ | ✓ |
| Long-horizon manipulation | ✗ | ✓ | ✓ |

表内脚注明确：`Original papers; cross = not demonstrated.` ✗ 代表原论文未展示，不代表理论上不能支持。这里不是我们把其他模块接入 G1 后的消融配置表；尤其不能混淆原始 LaViRA 与已加入 GATE 的 BaselineLAVIRA。

来源：[LaViRA 原论文](https://arxiv.org/html/2510.19655v1)、[UniLM-Nav 原论文](https://arxiv.org/html/2607.06537v2)，于 2026-09-26 核对。UniLM-Nav 的语义导航包括其集成的 HomeRobot 导航策略；完整 OVMM 的 find/pick/find/place 序列计入多阶段长程操作。其真实机器人为 B2+Z1，未展示人形全身操作。Training-free 仅限定交接模块。Ours 对照当前 GATE 稿件的 G1 T1–T4 及 OVMM 实验。详细口径也写入每页演讲者备注和 `verification.json`。

原版本完整保存在 `before_method_comparison/`。`build_options.py` 使用 `/usr/bin/python3` 运行，以匹配 `/tmp/align-ppt-deps` 中的依赖。

主交付：**GATE_teaser_three_styles.pptx**，共三页，每页一个版本：

1. A / Editorial white：白底、克制配色、细圆角比例条，推荐用于论文。
2. B / Warm cards：暖白底、衬线标题、并排数据卡片。
3. C / Navy rings：深蓝底、薄荷绿圆环，视觉对比更强。

每页都包含三个可以单独整组移动的原生 PPT 技能标记。选择窗格中名称为 `MOVE ME — Navigate`、`MOVE ME — GATE`、`MOVE ME — Manipulate`。点击边框或从选择窗格选取即可整组拖动；取消组合后可分别改图标、背景、文字。

Navigate 放在蓝线右侧；GATE 放在白色弯曲轨迹左侧；Manipulate 放在打开的垃圾桶上方。导航小箭头与蓝线方向一致。原始照片、相机小窗及蓝色轨迹保留，未使用生成式 AI。

右侧统计保持一致，w/o GATE 38/133 = 28.6%，w/ GATE 127/147 = 86.4%。统计口径与队列限制完整保留在每页演讲者备注中。新增能力对比与上方 w/o/w/ GATE 统计属于不同的比较。所有新增文字、比例条、圆环、勾叉、技能标记都是原生矢量对象，不是整页截图。

预览图 `preview-1.jpg` 至 `preview-3.jpg`，高清图 `figure-1.png` 至 `figure-3.png`，均由最终三页 PPT 实际导出；另留每版单独 PPT 方便后续编辑。主文件已核对三页、每页三个技能组合、全部数值、原照片字节一致、蓝线几何一致、画面无越界和丢失底部。
