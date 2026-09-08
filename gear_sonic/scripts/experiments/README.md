# ICRA 实验入口

这里提供 15 个入口，同名参数文件在 `gear_sonic/config/experiments/`。真实运行复用现有 `launch_inference.py` 的 tmux、控制网关和操作界面；**启动脚本后，按 `n` 才开始一次试验和计时**。脚本不自动重复、不自动切换条件。离线语义角色评测例外，它只处理给定图像集合。

本次实现包括简单几何接近/居中；不包括 **UniLM-Nav adapted to our platform** 和“定位 + 直接前往”。训练邻域覆盖率暂不计算。

## 先填写公共配置

配置按以下顺序读取：实验 YAML → 引用当前 `../launch_inference.yaml` → 合并本实验 `runtime_overrides` → 填入任务、条件和实验分支。相机、权重地址及底层控制参数来自同一份现有 agent 配置。每次实际启动保存一份不可随源 YAML 改动而变化的 `runtime.yaml`。

1. 编辑 `gear_sonic/config/experiments/tasks.yaml`。填写实际服务的 `policy_id`；核对 T1、T2 已有的训练字符串，填写 T3、T4 的 `vla_trained_prompt`。`policy_id` 是人工填写的检查点标识，脚本不能独立核实远端当前加载的权重。
2. 同文件中的 `vla_prompt` 描述真实物理目标，供 LA 的操作上下文和全部 VA 使用。`vla_trained_prompt` 保留训练时的原文，写入 VLA 初始配置及 `handoff_context`，启动、恢复和再次调用均使用它。导航路线仍单独传给导航规划器。
3. 编辑条件文件中的 `layout_id`、`start_id` 和各任务 `navigation_instructions`。表 1 的 `cases_main.yaml` 有 `visible_01` 至 `visible_08`、`invisible_01` 至 `invisible_08`；四个任务各自填写路线，共形成每任务 16 个模板。不可见条件要确认初始扫描也不可见；扫描后可见的情况单独记入 `scan_visible`。
4. 表 2 使用 `cases_nearfield.yaml`，填写起始距离、侧向偏移、方位角、yaw 偏差与起点分组。给定的 `nominal_01`、`distance_01`、`lateral_01`、`bearing_01`、`yaw_01` 是待填写、可复制扩充的模板，脚本不会自动摆放机器人。
5. 副表 1 使用 `cases_gates.yaml`。`normal` 不施加扰动；`perturbed` 必须填写扰动 `description`、`amplitude`、路线、布局和起点。每种门控的开关两组使用同一物理条件、扰动幅度和预算。需要不同门控的扰动协议时，复制条件文件并让对应的两个 YAML 引用它。

`null` 表示未知。权重标识、训练提示词、VLN 路线、布局、起点和启用扰动时的协议缺失会阻止真实启动；可选难度描述和位姿缺失会保留为空。不要用占位字符串冒充实际输入。

预算在 `tasks.yaml` 的 `defaults` 中统一设置，也可以在单个任务下面覆盖：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `total_timeout_s` | 600 s | 从 `n` 到试验结束的独立总时限，含模型等待和扰动准备 |
| `vla_timeout_s` | 180 s | 从首个有效动作发出起的 VLA 操作上限 |
| `fixed_manipulation_s` | 180 s | 关闭完成检查时的固定操作时长，不超过操作上限 |
| `recovery_budget_s` | 600 s | 未恢复成功时用于恢复耗时的惩罚值；总时限始终优先 |

原 agent 的导航步数、单段等待及操作检查窗口等预算保留在公共 runtime 配置中。调整任务预算时，对同任务所有方法使用相同设置；完成检查关闭组的固定时长也应一致调整。

## 启动命令

在仓库根目录运行。以下第一条只校验配置、列出必要缺项，不连接机器人或模型、不创建试验结果目录：

```bash
bash gear_sonic/scripts/experiments/full_vln.sh --task T1 --case visible_01 --dry-run
bash gear_sonic/scripts/experiments/full_vln.sh --task T1 --case visible_01
```

每个入口都支持 `--config /absolute/path/method.yaml`、`--task T1`、`--case visible_01`、`--dry-run` 和 `--output /absolute/path/results`。`--config` 接收实验 YAML，公共 agent YAML 通过其中的 `runtime_profile` 引用。相对引用以实验 YAML 所在目录为基准。

| 分组 | 脚本名（同名 YAML） | 变动 | 默认条件 |
|---|---|---|---|
| 表 1 | `full_vln.sh` | 完整 VLN agent | `visible_01` |
| 表 1 | `nav_direct_vla.sh` | 导航交接后直接 VLA | `visible_01` |
| 表 1 | `navila_basepose_vla.sh` | NaVILA 导航，共用 BasePose 与 VLA | `visible_01` |
| 表 1 | `geometric_vla.sh` | 头部单目标几何接近/居中 | `visible_01` |
| 表 1 | `full_objectnav.sh` | 完整 agent，目标搜索指令 | `visible_01` |
| 表 2 | `near_dual.sh` | 给定起点直接双相机 ALIGN → VLA | `nominal_01` |
| 表 2 | `near_head.sh` | 给定起点直接单头相机 ALIGN → VLA | `nominal_01` |
| 表 2 | `near_vla.sh` | 给定起点直接 VLA | `nominal_01` |
| 副表 1 | `gate_nav_on.sh` / `gate_nav_off.sh` | NAV → ALIGN 门控开启 / 关闭 | `normal` |
| 副表 1 | `gate_align_on.sh` / `gate_align_off.sh` | ALIGN → VLA 门控开启 / 关闭 | `normal` |
| 副表 1 | `gate_completion_on.sh` / `gate_completion_off.sh` | 完成门控开启 / 关闭 | `normal` |
| 副表 2 | `semantic_roles.sh` | 固定图像集合离线评测 | `dataset` |

真实启动使用原有外部 VLA 策略服务、相机与机器人启动前提。脚本不训练模型、不替换远端权重。近场与 NaVILA 分支不启动本地 NavDP 模型及规划器；它们仍使用共同的传感器、控制网关和速度执行器。原 tmux 启动器会清理并重建现有 `sonic_inference` 会话。

在 tmux 的操作面板中：

- `n`：新建试验 ID 并开始；当前试验未结束时再次按 `n`，先取消并关闭旧试验，再开始新试验。旧回复无法控制新试验。
- `g`：人工确认当前试验已成功，记录按键时刻及从 `n` 开始的完成耗时，并写入成功、全部子目标完成及适用的导航成功标注。立即结束本轮 agent，回到 PLANNER 站立，C++ 控制循环继续运行；迟到的 VA 回复不能覆盖成功结果。没有运行中的 agent 时忽略此键，`s` 仍用于手动后退。
- 空格：取消、停止并记录人工接管。若物理任务已成功完成，应在标注中填写停止前的实际完成时间；完成后的停止不降低 SR。
- 扰动组出现 `PERTURBATION_CUE` 时，施加预先定义的扰动，立即输入 `:perturb` 并回车。系统记录该时刻并继续阶段检查；提示前、重复或无扰动组的标记会被拒绝。此标记是操作员确认的触发时刻，可用外部视频核查。
- 扰动等待计入总时间。预定义扰动不算额外接管；额外切换控制模式、手动改提示词、暂停策略等单独记录为人工干预。

## 特殊分支的执行规则

所有方法在 MANIPULATE 操作阶段的 VA 检查均使用头部 `ego_view` 图像，包括 UNKNOWN 重试和完成门控关闭时的旁路检查；日志中的 `camera` 记为 `head`。

**单头相机**：角色选择、检测/跟踪、BasePose 控制、ALIGN 视觉交接仅使用 `ego_view`。VLA 仍使用头、胸、左右腕四视角；首帧归档中的胸部图像和离线 mask 不参与该版本的对齐决策。近场导航阶段记为“不适用”。

**简单几何对齐**：参数在 `tasks.yaml` 的 `geometric` 中，目标由每任务 `geometry_target` 给定。头部 RGB-D + YOLOE，先以 `wz=0.3 rad/s` 转向目标中心，再以 `vx=0.4 m/s` 接近，最后以 `wz=0.2 rad/s` 居中。纵向距离默认 1.1 m、距离容差 0.1 m、角度容差 9°，连续 3 个新观测满足才记为 aligned。目标丢失时保持静止，过近时以相同速度后退纠偏，无侧移及结构边缘 yaw 控制。60 s 超时记录 `geometric_timeout`、停止底座，完成共同的交接检查记录后继续 VLA；VA 建议不会阻挡此分支。硬件/控制错误、取消和总时限仍可结束试验。对齐超时不自动等于物理任务失败。

**NaVILA**：在 `tasks.yaml` 的 `navila.host` / `port` 填写可达的模型 ZMQ 端口；默认 `127.0.0.1:30000` 只是客户端配置，可以配合已有端口转发。不要填写 SSH 端口。需要认证时设置 `NAVILA_API_TOKEN`。每次 `n` 先 `ping`、`reset`，同一试验导航恢复保留服务端历史。请求使用 MessagePack `endpoint=get_action`、JPEG、instruction、sequence，响应检查字典类型、协议版本、序号和错误。支持 25/50/75 cm 前进、15/30/45° 左右转和 stop；超时或不支持的动作结束试验，绝不猜测动作。离散动作由共享网关定时执行并停止。使用专用服务实例，避免其他客户端修改同一服务端历史；本次没有修改服务端。

**门控关闭**：仍保存 VA 观测和建议，与实际执行决定分开。NAV 门控关闭时，只在最终目标的导航结束候选后放行，中间探索步不冒充阶段结束。ALIGN 门控关闭时，控制器到达终止状态后放行；尚未找到角色、尚未启动控制器的状态仍重试。完成门控关闭时，VA 在旁路运行，执行到首个有效动作后固定 180 s 停止并记录完成候选；阻塞或迟到的 VA 回复不延长操作时间，迟到结果不再计入有效门控决定。所有分支保留底层保护、取消和总时限。

## 日志、首帧与人工标注

每次脚本启动生成 `outputs/experiments/<方法>_<任务>_<条件>_<会话编号>/`：

```text
runtime.yaml                         # 完整有效配置、版本标识、预算和两个 prompt
events.jsonl                         # 本次启动的所有试验与追加标注
checks/<trial_id>/*.png               # 每次 VA 实际检查的图像
snapshots/<trial_id>/vla_<skill_id>/   # 每次 VLA 首次真正请求策略时的输入
  ego_view.jpg / chest_view.jpg / left_wrist.jpg / right_wrist.jpg
  *_depth.npy / snapshot.json
  masks.json / *_roles/*_mask.png     # 试验后离线生成
semantic/sample_*/                    # 副表 2 的角色 mask
```

`events.jsonl` 使用试验 ID、事件 ID 和技能 ID 关联以下证据：试验开始/结束、各阶段开始/结束、VLA 请求/确认/首次推理/首个动作/停止、对齐状态和误差、VA 轮次/视角/检查图像/观测时间/回复时间/置信度/简短依据/角色、合并门控结果、扰动提示与确认、取消/接管、首帧和缺失原因。配置记录方法、任务、条件、策略/模型标识、两个 prompt、预算、代码版本及有效配置摘要。重试按失败后的恢复统计；正常探索移动不算重试。

相机源时间戳可获得时保存；不可获得时明确为空，另保留本机取得观测的时间。四张首帧 JPEG 是真正交给策略的原始字节。头/胸深度按源时间戳匹配，默认最大误差 5 ms；匹配失败不使用后续帧补位。保存深度单位/来源、相机内参与安装标定、可获得的 FAST-LIO 位姿与坐标系属性。配置外参不是每帧测量的全身运动学外参，缺失机器人位姿时不能凭它恢复世界坐标。

运行中不额外检测首帧角色。结束 tmux 会话后启动器尝试离线补 mask；若只是 detach，需结束会话后手动执行：

```bash
.venv_inference/bin/python -m gear_sonic.experiments.offline /absolute/path/run/runtime.yaml
```

补 mask 只读取保存的首帧。优先采用实际对齐角色；纯 VLA 等未选角色的分支用 `vla_prompt` 离线选择，并标记 `offline_semantic_selection`。几何分支标记其配置的居中目标。文件齐全的快照可重复补处理，已成功生成的记录跳过。没有终止记录的试验会阻止补处理，需要先确认真实运行已停止并关闭该记录。

最终物理成功、有效子目标数、实际完成时间、Nav SR，以及每次门控的独立真值均需要人工标注。**VA 宣告完成不会自动填成成功，也不会替代实际完成时间。** 开始前固定任务成功的保持时间、倾倒条件和放置区域。

运行中按 `g` 就是一次人工成功标注，汇总可直接读取；也可用下方命令追加或更正标注。

```bash
# 查看 trial_id、门控 event_id 和语义 sample_id。
.venv_inference/bin/python -m gear_sonic.experiments.results inspect /absolute/path/run/events.jsonl

# 从按 n 起算的实际物理完成时间，单位秒；progress 填有效子目标个数。
.venv_inference/bin/python -m gear_sonic.experiments.results result /absolute/path/run/events.jsonl --trial TRIAL_ID --success true --progress 2 --completion-time 73.4 --nav-success true

# 已失败试验；不需要填写完成时间。需要排除时另外填写 --exclude-reason 原因。
.venv_inference/bin/python -m gear_sonic.experiments.results result /absolute/path/run/events.jsonl --trial TRIAL_ID --success false --progress 1

# 独立判定该门控事件观察时刻是否满足就绪/完成条件。
.venv_inference/bin/python -m gear_sonic.experiments.results gate /absolute/path/run/events.jsonl --event GATE_EVENT_ID --truth false
```

标注追加到原日志，重复标注以最新值为准。已知成功但缺少完成时间可只填成功，耗时保持缺失。`--condition-file /path/verified_condition.yaml` 可追加经人工核实的布局、起点、扫描可见性和路径距离等条件信息，汇总时覆盖配置描述。异常断电/退出后，确认运行已经停止，可在 `result` 命令加 `--close-interrupted` 补终止记录；它只关闭日志，不是机器人停止命令。此时自动运行时长保持缺失，并注明终止记录为人工补记；论文耗时仍采用人工完成时间或失败惩罚值，不拿离线补记时刻作为实际停止时间。

默认不归档逐帧控制调试、完整模型请求和连续视频。tmux 保留原有实时显示；独立标注所需的外部视频由实验人员按统一协议采集。

## 固定图像的语义角色评测

离线工具同样读取进程环境或仓库 `.env.local` 中的 `LAVIRA_VA_API_KEY` / `DASHSCOPE_API_KEY`，只调用 VA，不需要另外配置 LA 密钥；密钥不写入结果文件。

将 JSONL 清单按 `semantic_samples.example.jsonl` 填写；相对图像/深度路径以清单所在目录为基准。每个样本包含 `sample_id`、`task_id`、`vla_prompt`、`rgb`；几何可用性还需要同步的 `depth`（二维 uint16 `.npy`）、`camera_info`（内参、`depth_scale_m`、来源）、`mount_calibration`（安装角度和偏移）。示例仅说明格式，不附带可用于评测的图像或标定值。

```bash
bash gear_sonic/scripts/experiments/semantic_roles.sh --task T1 --manifest /absolute/path/samples.jsonl --dry-run
bash gear_sonic/scripts/experiments/semantic_roles.sh --task T1 --manifest /absolute/path/samples.jsonl
.venv_inference/bin/python -m gear_sonic.experiments.results semantic /absolute/path/run/events.jsonl --sample SAMPLE_ID --position true --yaw true --joint true --usable false
```

每次只运行 `--task` 所选任务的样本。检测状态和几何可用状态自动记录；位置/yaw/联合角色正确和检测实例/几何是否真正可用由独立标注给出，允许多个合理答案。缺少深度或标定时，自动几何状态为空并记录原因。可以只标注部分字段（例如省略尚无法判断的 `--usable`），以后追加；缺失真值与已知不可用分开统计。汇总同一个固定集合时只选一次评测记录，跨输入日志重复的 `sample_id` 会报错。

## 汇总

```bash
.venv_inference/bin/python -m gear_sonic.experiments.results summarize /absolute/path/run1/events.jsonl /absolute/path/run2/events.jsonl --output /absolute/path/tables
# 也可传包含多个运行目录的父目录；只合并同一套冻结条件与权重的结果。
```

输出 `trials.csv`、`statistics.json`、`tables.md`。Markdown 对应表 1、表 2、副表 1、副表 2；JSON 另有逐任务、可见性、近场起点分组、Nav SR、交接率、VLA 启动率、条件操作 SR、重试和数据完整性统计。

- VLA 启动以首个有效动作实际发出为准，恢复同一技能不重复计数；请求和 ACK 单独保留。条件操作 SR 的分母为真正启动 VLA 的试验。
- SR 使用独立成功标注；报告分子/已标注分母、样本量、缺失数和 Wilson 95% 区间。四任务宏平均等权；宏平均 SR 区间采用固定随机种子的任务分层 bootstrap（10,000 次），样本很少或全成功/全失败时需结合逐任务 Wilson 区间解释。
- 失败按同任务 `total_timeout_s` 计入平均耗时，成功使用人工完成时间。缺少成功时间时不拿系统结束时间补值，相关耗时平均也保持缺失。额外接管发生在物理完成前时不计自主成功。
- Progress 按有效子目标数 / 任务子目标总数计算，依赖关系由标注员判断。先逐任务平均再宏平均。
- 门控错误率只使用该实验所消融的门控、独立标注的有效检查事件。无扰动且真值满足构成错误阻挡率分母；已确认扰动且真值不满足构成错误放行率分母。迟到回复保留证据但不再产生有效决定。扰动没有实际确认的试验不进入扰动恢复分母。
- 无有效分母显示“不适用”，缺少标注显示“待标注”。四任务未齐不生成三任务等冒充的宏平均；排除记录保留原因，不进入指标。训练邻域覆盖率显示“未计算”，首帧深度/位姿/mask 的缺失单独记录。

## 离线验证

新增行为测试位于 `gear_sonic/tests/test_experiments_runtime.py` 和 `test_experiments_evidence.py`，使用假模型、模拟控制状态、MessagePack 协议响应与已知统计记录。真实硬件、相机时序、远端模型可达性和实际操作成功率仍需在填好条件后，由操作员按 `n` 验证。
