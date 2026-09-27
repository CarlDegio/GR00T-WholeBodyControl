# UniLM-Nav adapted to our platform

本分支实现 `experiment.md` 表 1 的 UniLM-Nav 交接方法。完整 agent 与独立入口使用同一个 BasePose backend。它输出并执行世界坐标系中的站位，而不是重新向 NavDP 提交一个距离容差为 1.5 m 的导航点。

## 运行入口

完整 agent：

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_inference/bin/python /home/user/Project/GR00T-WholeBodyControl-unilm-nav/gear_sonic/scripts/launch_inference.py --config /home/user/Project/GR00T-WholeBodyControl-unilm-nav/gear_sonic/config/launch_inference.yaml
```

在 tmux 的 control 面板按原流程用 **K** 开启控制，确认处于 **PLANNER（O）** 模式，然后按 **N**。执行顺序为：原 LaViRA/NavDP 导航 → 最终目标导航交接检查通过 → UniLM → 到位停稳 → 原 VLA → 原完成检查。

独立 UniLM，从当前近场位置开始：

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_inference/bin/python /home/user/Project/GR00T-WholeBodyControl-unilm-nav/gear_sonic/scripts/launch_unilm_nav.py --config /home/user/Project/GR00T-WholeBodyControl-unilm-nav/gear_sonic/config/launch_inference.yaml --task 'collect the pill bottle'
```

同样开启控制并确认 PLANNER 模式，然后按 **B**。到位停稳后结束，不启动 VLA 操作任务。省略 `--task` 时读取配置的 `components.vla.vla_prompt`；`--task` 只覆盖 UniLM/agent 的通用任务，不改 VLA 训练提示词。加 `--dry-run` 只打印有效配置，不启动进程。脚本会将独立模式的完整配置写入本分支 `outputs/runtime_profiles/`。

两个入口都用 **Space** 取消当前任务。完整 agent 运行期间 B 被 Gateway 拒绝，不抢占任务。完整运行结束后也可直接按 B 做一次独立定位。每次 B 都重新采集五组观测，不复用上一任务的图像。已有 `sonic_inference` 会话时，独立启动器拒绝覆盖它，应在已有会话中使用 B。

独立模式启动 SensorGateway、ControlGateway、FAST-LIO/Livox、Planner Executor、SONIC、操作控制台与现有 VLA 服务。VLA 服务保留 PLANNER 控制消息转发，因为它负责连接 C++ 控制通道。独立模式不启动 NavDP、LA/VA、Depth Anything，也不等待 VLA 策略服务器就绪；VLA 服务自身可能打印策略服务器不可达的提示，但 B 定位不调用该策略。

## 实验配置

`components.unilm_nav.enabled: true` 选择 UniLM；改成 `false` 即恢复原 BasePose backend 和原 agent 对齐交接流程。原 BasePose 参数保留。

**正式表 1 对比使用 VLN。** 当前导航仍是搜索手推车上的纸箱，通用操作任务已改为接近桌上的蓝篮子、拿起药瓶并放入篮子，两者不匹配，不应将这个默认组合直接用作正式评测。每个配对实验需使用与 Full agent 相同的：

- `components.lavira.navigation_mode: vln`、路线 `mission`、最终 `global_target`。
- `components.vla.vla_prompt`：UniLM 三阶段推理、LA 规划、对齐与操作完成检查，以及默认数据记录任务使用的通用指令。
- `components.vla.vla_trained_prompt`：仅用于 VLA 策略推理，当前保留 `collect the pill bottle`。
- 起点、场景、物体布局、VLA 权重及操作预算。UniLM 到位后启动相同 VLA，不额外引入对齐视觉放行判断。

中间路标导航不会触发 UniLM。只有最终导航目标匹配且原导航交接检查通过时才进入 UniLM；失败直接结束本轮，不由 LA 重新规划另一轮站位。

两份提示词只在 `components.vla` 配置，LaViRA 直接读取同一份 `vla_prompt`，无需再设置 `components.lavira.manipulation_prompt`。`start_vla_task` 的 `task` 仅作为 agent 上下文，启动或恢复 VLA 时不会覆盖 `vla_trained_prompt`。显式留空的通用提示词不会回退到导航 mission 或训练提示词；UniLM 沿用原有的空任务拒绝逻辑。

## 观测、推理与执行

1. 完整模式在每个 NavDP 平移导航段中以 2 Hz 独立保存最近五组胸相机 RGB-D 与同步位姿，收到该段 `reached` 后冻结；切换导航目标时保留前段历史。对齐使用本轮从开始到来源导航段为止、所有已到达导航段保存的观测，按时间排序并去重，合并后可以超过五组。末段很短甚至没有新帧时，可使用前段历史；合计不足五组或来源段未到达/不匹配仍报告失败。图像来自 UniLM 独立传感器采集，不读取 LA 五视角图库；导航开始时仍可用独立采集通道的最新有效帧作种子，导航后的交接检查或原地朝向调整不会替换冻结图像。新任务、取消或 generation 切换时清空历史。独立模式在 B 启动后仍采集五组新观测。
2. 第一次模型调用选择一个视角；第二次在该图像中定位任务交互点；第三次使用标记交互点的 RGB、深度反投影后的三维点和 G1 几何，预测相对站位 `(x,y,theta)`。正常流程恰好三次调用。每次调用的网络或 JSON 格式错误最多重试两次，共享整轮 180 s 时限。
3. Qwen 的交互点使用严格的 `0 < u,v < 1000`，先换算到图像像素，再从所选历史帧的对应深度反投影。局部站位通过该历史帧的位姿变换为固定世界目标；执行时不会把它误当成当前机体系坐标。
4. 执行朝向由 `atan2(target_y-y, target_x-x)` 计算，模型输出的 theta 只保留作诊断。BasePose 通过原 Gateway → Planner Executor → VLA 的 PLANNER 转发 → SONIC 执行。FAST-LIO 位置误差首次小于 5 cm 时锁定本轮位置已到达。此后位置只写入诊断日志，最终转向和停稳均不再检查距离，也不再退回接近模式；新任务重新判断位置到位。
5. 采用分段转—停—直走—停控制。首次使用当前位姿计算接近方向；转向误差严格小于 7° 时开始发送全零速度，保持 1 s（`approach_hold_s`）。仅当最后 0.3 s（`approach_stable_s`）的所有控制采样均在朝向容差内，才进入前进；若未通过则继续调整方向，并重新执行完整的 1 s 零速度检查。直走以 20 Hz 发送 `(0.4, 0, 0)`，每段最多 1 s（`walk_duration_s`），走的时候不发转向指令。随后全零速度保持 0.5 s（`walk_pause_s`），以该等待结束时的 FAST-LIO 位姿重新计算并固定下一段接近方向，再次转向及停稳确认后才走。任意阶段检测到位置误差首次小于 5 cm，均在同一控制周期立即发送全零速度，并保持 1 s（`position_settle_s`），随后才调整最终朝向；等待期间的位置抖动不会重新前进或重置计时。每次转动的非零角速度至少 0.1 rad/s，最大 0.3 rad/s。取消、传感器超时和底层停止仍可立即打断执行。
6. FAST-LIO 实际朝向与最终目标的误差在 7° 内后，验证朝向停稳 1 s。停稳期间实际朝向超差只重置朝向稳定计时并继续转向，不重新检查位置。SONIC heading lag 仅作诊断，不参与转向速度、前进放行或完成判断。完整 agent 随后直接进入原 VLA readiness / 启动确认流程；独立 B 模式仍在此结束。

配置默认值：

| 项目 | 值 |
| --- | --- |
| 模型 / 思考模式 | `qwen3.8-flash` / 关闭 |
| 相机 | `chest_view` / 对齐原始深度 `camera/chest_view_depth` |
| 相机离地高度 | 1.0 m |
| 安装 pitch / roll / yaw | −3° / 0° / 0° |
| 水平前后 / 左右偏移 | 0 / 0 |
| RGB–深度最大时间差 | 5 ms |
| 相机–FAST-LIO 位姿最大时间差 | 100 ms |
| 当前里程计 / SONIC 朝向新鲜度 | 300 ms |
| 控制频率 | 20 Hz，沿用 `base_pose.planner_hz` |
| 前进速度 / 上限 | 0.4 / 0.4 m/s |
| 非零转向速度 | 0.1–0.3 rad/s；对准时为 0 |
| 起步前朝向确认 | 全零速度保持 1 s，最后 0.3 s 朝向误差始终 <7°；未通过则重调并重新确认 |
| 单段直走 / 段间等待 | 最多 1 s，角速度为零 / 全零速度保持 0.5 s 后重新计算接近方向 |
| 到位后等待 | 全零速度保持 1 s |
| 位置模式切换 | 首次 <0.05 m 立即发零速度并锁定到位；等待 1 s 后最终转向，不再检查位置 |
| 完成条件 | 已进入最终转向后，实际朝向误差 ≤7° 并停稳 1 s；无需最终距离复核 |
| 全轮时限 | 180 s，含采集、推理重试、跟踪和停稳 |
| 预测及接近段距离保护 | 到位前相对当前站位超过 3 m 则失败；不裁剪预测点；到位后距离仅作诊断 |

`unilm_nav.camera: chest` 同时选择 `base_pose.dual_chest_*` 中的 RGB、深度流和外参；高度由 `unilm_nav.camera_height_m` 指定。内参和深度单位读取所选胸相机帧的 Gateway 元数据，深度必须对齐到 `chest_view`。此选择同时作用于完整 agent 的导航历史和独立 B 的新观测。配置也支持 `head`，对应原头相机流及外参，切换时需同步设置该相机的离地高度。

控制器执行接近方向对准、起步前停稳确认、分段直走、段间等待、到位等待和最终转向，直接使用 FAST-LIO 实际位姿的朝向误差。转向速度为 `sign(error) * clip(1.5 * abs(error), 0.1, 0.3)` rad/s；走和等待期间角速度均为零。SONIC heading lag 仅作诊断。运动日志保留 heading lag、`approach_heading_rad`（本段固定接近朝向）、`position_acquired` 和 `phase`，区分接近（`approach`）、起步确认（`approach_settling`）、直走（`walk`）、段间等待（`walk_pause`）、到位等待（`position_settling`）、最终转向、朝向停稳和完成。各等待分别计时。`reached/aligned` 表示上述流程完成，不保证最终距离仍在 5 cm 内。取消、超时、过期传感器数据或跟踪失败会发送停止，晚到的模型结果不能重新驱动机器人。沿用原速度边界和底层保护，**没有增加在线绕障规划或点云路径碰撞检测**；当前位姿跟踪器不具备避开途中障碍的能力。这属于平台执行适配的限制，评测需如实记录。

## 坐标与 robot configuration

论文的 `robot configuration` 在此指供模型推断可达站位的机器人几何描述，不是 API 配置文件。删除它会失去方法中的机器人几何约束。本实现提供 G1 肩部位置、肩到腕的运动链长度、手部几何范围、名义最大臂展和身体投影范围；不输出 Stretch 式的 arm extension / lift height。操作关节运动全部交给原 VLA。

几何从本分支 `g1_29dof_with_hand.urdf` 及其碰撞网格读取，手部网格范围作为 dex1 操作空间的名义代理，并非在线 IK 可达性验证。以地面上的骨盆投影为局部原点，X 前、Y 左、Z 上。胸相机固定离地 1.0 m，pitch 为 −3°，其余安装外参沿用 0。机器人站姿几何仍以原头部安装点离地 1.1 m 为参考，骨盆及肩部名义高度保持原值；相机高度仅用于当前观测的投影和模型中的相机描述。

FAST-LIO `/Odometry_loc` 的 `child_frame_id=body` 是 MID360 IMU 坐标系。代码利用标准 URDF MID360 安装位姿及现有 `mid360.yaml` 的 LiDAR–IMU 平移 `[-0.011, -0.02329, 0.04412] m` 转换为骨盆位姿，再使用平面 XY/yaw。不能将 IMU 位姿直接当成机器人底座位姿。RGB-D 与 ROS 时间戳按各自源时间配对，要求两端时钟同步；不允许用新的位姿替代历史图像的位姿。

## 环境与密钥

所有启动命令明确使用 `/home/user/Project/GR00T-WholeBodyControl` 下已有的对应虚拟环境，SONIC 权重也引用该目录的已有文件。启动器设置工作目录和 `PYTHONPATH` 为当前分支，避免共享虚拟环境的 editable install 导入原分支代码。`SONIC_ENV_ROOT` 可覆盖环境根目录。

C++ 使用 `components.deploy.onnxruntime_root` 指定的已有 SDK，当前为 `/home/user/Project/.tools/onnxruntime`。`deploy.planner` 和 `deploy.motion_data` 也显式引用原项目资源，避免新分支缺少 Git 忽略的 planner 模型。启动预检同时检查这些文件。C++ 构建和运行输出持续保存在本分支 `outputs/logs/inference/deploy.log`；启动器等待其 5557 输出端口就绪后才继续，最长等待时间由 `deploy.ready_timeout_s` 控制，默认 600 s。

2026-09-08 的两次完整启动在 C++ 构建阶段失败，新构建目录缺少原项目 CMake 缓存中的 ONNX Runtime 路径。已修复，并编译出本分支的 `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`。启动器现在检查实际进程是否退出，不再把停留在 shell 提示符的面板当成正常部署。5557 端口就绪仅表示 C++ 初始化完成；实际状态帧仍要在按 K 开启控制后产生。

UniLM 与 LA/VA 使用相同的 OpenAI-compatible API 地址和配置方式。启动器加载原项目 `.env.local` 中的 `DASHSCOPE_API_KEY`，也支持显式 `UNILM_NAV_API_KEY` 覆盖；密钥不复制进本分支，也不写入请求日志。

独立模式启动就绪检查只验证本机依赖、所选胸相机 RGB-D 和同步 FAST-LIO 位姿。物理定位还要求 K 开启后的有效 SONIC 状态；没有有效状态就失败停止。`--dry-run` 不代表硬件就绪。

## 日志与验证

每轮在本分支 `outputs/unilm_nav/<timestamp>_g<generation>_s<skill>/` 保存：

- `config.json`：操作任务、来源导航段、模型、相机与控制参数。
- `observation_*.npz`：RGB、深度、内参、深度单位、同步历史位姿与时间戳。
- `view_*.png`、`grounding_*.png`、`pose_*.png`：三个阶段发送的图像。
- `*_attempt_*.json`：完整模型请求、响应、重试错误和阶段调用耗时。
- `prediction.json`：交互点、三维点、模型位姿、几何朝向、参考历史位姿和世界目标。
- `motion.jsonl`：逐步实际位姿、速度指令、平移/朝向误差、heading lag。
- `result.json`：终态、总耗时、失败原因、目标和最后实际跟踪状态。

已验证三阶段调用、坐标尺度、胸相机 1.0 m / −3° 及头相机 1.1 m / −38° 外参、历史位姿变换、Gateway RGB-D/位姿配对、完整 agent 末段交接、独立五帧采集、重复启动、任务抢占保护、取消、过期结果、超时、数据丢失，以及带朝向滞后的模拟反馈到点控制。

2026-09-09 的控制更新另覆盖 10/15 cm 严格边界与回差、最低转速、0.3–0.5 s 前进脉冲、脉冲中途取消、最终转向完成后的距离漂移，以及原 VLA 就绪检查继续生效。UniLM 模块 48 项测试通过，连同 LaViRA、RuntimeConfig、ControlGateway 和 PlannerExecutor 共 158 项通过。对 2026-09-08 故障的 837 帧固定观测回放，位置状态由原来的 19 次边界切换变为一次进入最终转向；这不代表新指令下的真机闭环轨迹已经验证。

同日切换胸相机后，新增验证两路相机同时存在时 RGB、深度、内参及深度单位的配套选择、胸相机外参投影、误用头相机深度的拒绝，以及站姿几何和里程计变换保持原值。UniLM 共 52 项测试通过，连同 LaViRA、RuntimeConfig、ControlGateway、PlannerExecutor、启动器和 SensorGatewayClient 共 207 项通过。该次修改未启动真机运动；运行进程需重启后加载新的相机配置。

同日 20:05 开始的真机轮次 `1788955511645022721_g2_s2` 在第 24.1 秒首次进入 10 cm，随后因转向期间距离越过 15 cm 回退接近 17 次，最终耗时 162.5 秒完成。针对这轮问题，控制器现改为首次位置到位即锁定，覆盖短步、暂停、最终转向和朝向停稳全过程。移除 `position_reapproach_m` 配置；到位后的距离只作诊断，取消、传感器有效性检查和全轮时限仍生效。重启时使用当前 `launch_inference.yaml` 重新生成运行配置。

本次相关回归 183 项通过，覆盖转向期间位置漂移、短步及暂停期间到位、朝向停稳受扰后的重新计时、到位前距离保护，以及到位后的取消、超时和传感器丢失。对上述轮次的 3183 帧固定观测回放，回退接近由 17 次变为 0 次，进入最终转向后不再输出前进指令。回放结果保存在 `outputs/diagnostics/unilm_position_latch_20260909/`。本次没有启动真机运动；固定观测回放只验证控制决策，不预测新指令下的实际轨迹或完成时间。

同日 22:48 的轮次 `1788965300453520771_g1_s2` 在第 58.6 秒锁定位置，随后最终朝向误差长期停在 8–10°，第 143.8 秒由操作台取消。原控制将实际朝向误差减去 heading lag，后段小误差在最低角速度限幅下反复变号。现改为只用实际朝向误差控制转向、前进放行和朝向停稳，heading lag 保留诊断。UniLM、PlannerExecutor、ControlGateway 和 LaViRA 共 170 项测试通过，覆盖设定朝向越过目标时不反转、速度上下限，以及实际朝向已对准时的前进和完成判断。2762 帧固定观测回放中，第 70 秒后的 1183 次指令反向变为 0 次，最终转向期间没有前进指令；记录位于 `outputs/diagnostics/unilm_actual_yaw_20260909/`。原观测仍未满足连续 1 s 的实际朝向容差，因此回放不报告完成，也不代表新控制已通过真机闭环验证。

初版相关回归共 258 项通过，当时新增模块的 Ruff 检查通过。测试使用原项目含 pytest 的 `.venv_teleop`，完整回归运行方式如下：

```bash
cd /home/user/Project/GR00T-WholeBodyControl-unilm-nav
PYTHONDONTWRITEBYTECODE=1 /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -B -m pytest -q \
  gear_sonic/tests/test_unilm_nav.py gear_sonic/tests/test_lavira_agent.py \
  gear_sonic/tests/test_control_gateway.py gear_sonic/tests/test_launch_tmux_panes.py \
  gear_sonic/tests/test_base_pose_yolo_adapter.py gear_sonic/tests/test_base_pose_dual_yolo_adapter.py \
  gear_sonic/tests/test_runtime_config.py gear_sonic/tests/test_runtime_snapshot.py \
  gear_sonic/tests/test_sensor_gateway_client.py gear_sonic/tests/test_planner_velocity_executor.py \
  gear_sonic/tests/test_run_sensor_gateway.py
```

2026-09-07 使用合成图像调用实际 `qwen3.8-flash` API，完成三阶段推理。最终版本联调中视角选择、交互点定位首次成功；站位阶段第一次返回 JSON 数组，格式校验拒绝后重试一次成功，共四次 API 请求。该测试验证模型服务、输出协议及格式重试，不代表真实场景识别成功率。记录位于本分支 `outputs/unilm_nav/api_smoke_1788779189690953856/`。本次没有启动真机运动，未进行端到端真机验收。

真机验收先从近场独立 B 定位开始：确认传感器就绪，检查预测世界点与实际停止位姿，测试 Space 取消；再在配对 VLN 配置下执行 N，确认中间路标不触发、最终交接只执行一次、到位后原 VLA 接管，并检查失败时是否停止。

## 与论文的对应关系

依据 [UniLM-Nav v2 附录 A，图 5–7](https://arxiv.org/html/2607.06537v2) 公开的 pick/place 提示模板实现三阶段语义流程。模板已公开；检查时[项目页](https://unilm-nav.github.io/)尚未提供可用的官方代码入口。本实现属于方法复现及平台适配，不宣称运行了作者原始代码。

保留视角选择、任务相关 grounding、深度提升、几何辅助站位，以及正文和附录 B.2 中的几何朝向计算。公开的 pick/place 两套模板合并为三个任务条件化提示；对于本项目复合任务，只定位首次交互，后续动作由同一个 VLA 完成。原论文没有直接评测本项目的抽屉等所有复合任务，应在实验中说明这个任务适配。

原方法的真实平台为 B2+Z1、腕部 Orbbec 相机、LIO-SAM、Nav2 位姿执行，并有操作阶段的视觉细化。本分支使用 G1、胸相机、FAST-LIO、SONIC 和共享 VLA，不复制其腕部操作控制。原文真机使用 Gemini 3 Flash Preview，未给出真机多模型排名；本实验按约定使用 Qwen 3.8 Flash，应作为模型适配差异报告。
