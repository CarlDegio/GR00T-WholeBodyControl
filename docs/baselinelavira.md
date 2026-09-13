# baselinelavira

本实现只在独立 Git 分支 `baselinelavira` 中维护，工作目录为
`/home/user/Project/GR00T-WholeBodyControl-baselinelavira`。
原目录 `/home/user/Project/GR00T-WholeBodyControl` 继续使用 `agent_full`，
baseline 的代码、配置、启动入口和公共模块接入已迁出。
新分支继承迁移时相机、对齐和 Sonic 的公共代码状态；之后两份源码及 YAML 独立修改。
本机虚拟环境、凭据文件、模型及已编译 Sonic 资源通过忽略的符号链接复用，
iPlanner 源码和 checkpoint 仍使用下面列出的外部路径。
新实验默认写入新工作目录的 `outputs/experiments/`；历史实验数据仍在原目录。

在当前 Groot 流程中，将导航部分替换为本地
`uni-lavira-code/real-world-code/unitree_g1` 的 LaViRA 方法：

```text
现有相机 → LaViRA LA 方向/策略 → LaViRA VA 自主选择导航物体/区域
        → bbox 底边中心 + 深度 → iPlanner → Pure Pursuit
        → 现有 Planner Executor → SONIC
```

LA 根据原版导航提示词维护 TODO、选择方向和策略；VA 根据全局目标和策略
独立选择下一步导航物体或区域。全局目标可见时优先选择它，否则选择策略
中的地标/区域。MOVE_TO 不再先调用原来的“指定物体 grounding”来生成目标。
原版依然输出 bbox；区别在于导航目标由 VA 自主选择。

导航提示词保留原版策略和选点规则，JSON 输出示例由适配层统一提供：VA 必须
返回数值型 `confidence`（0–1），缺失、字符串、百分数或越界值不会被补成
默认置信度。校验重试会明确指出字段问题。LA 的 `global_target` 必须是导航
终点的物体/区域名称，不能直接复制整句路线或后续操作指令。实验模式下，
导航 LA/VA 的原始回复及校验错误保存在 `lavira_requests/responses/`，成功和
失败的回复都保留；这与原有请求文件分开存放。

LA 当前视图使用 `CURRENT_VIEW_1` 至 `CURRENT_VIEW_5` 明确标注，
`chosen_view_index` 必须与 `turn_direction` 的实际角度一致，否则重试。
历史 MOVE_TO 图片标为 `PAST_MOVE_TO_*`，附带控制结果和观测结果；它们仍是
旧位置的结果照片，不是原版的轨迹历史图，也不能参与当前视图编号或方向选择。
VA 明确接收转向后的前视图，策略中的左右描述属于转向前的全景坐标。
目标、相关地标及有效通道都不可见时，允许 `REOBSERVE` / `bbox_2d: null`，
沿用现有未找到目标的交接检查和后续观察流程；不能为凑 bbox 选择无关桌子。
`STOP` 需要可见全局目标的有效框。这些输出约束属于本适配层的修复。

`baselinelavira.py` 每次从当前 `gear_sonic/config/launch_inference.yaml`
读取配置，再叠加 `gear_sonic/config/baselinelavira.yaml`。只新增
`navigation_policy.backend: iplanner`、iPlanner 服务和导航控制参数；主配置不被改写。
当前相机选择、RGB-D 对齐、深度来源、全景转向、BasePose 几何对齐、
VLA、FAST-LIO 及底层 SONIC 均复用现有实现。导航改用原版 LA/VA 提示词，
模型端点和凭据仍沿用当前配置；ALIGN/MANIPULATE 的提示词及交接流程保持。

与 `full_vln` 相同的实验启动入口是
`gear_sonic/scripts/experiments/baselinelavira.sh`。它读取
`gear_sonic/config/experiments/baselinelavira.yaml`，复用公共 `tasks.yaml`、
`cases_main.yaml`，并按任务和条件填入路线、全局目标、操作语义、训练 prompt、
策略标识和预算。相机及对齐配置仍继承当前 `launch_inference.yaml`；实验 YAML
通过 `runtime_overlays` 引用上述 baseline 配置，再应用 `runtime_overrides`。

从独立工作目录运行：

```bash
cd /home/user/Project/GR00T-WholeBodyControl-baselinelavira

# 检查任务/条件配置和 iPlanner 本地文件，不启动服务、不创建实验结果。
bash gear_sonic/scripts/experiments/baselinelavira.sh --task T1 --case visible_01 --dry-run

# 配置和权重就绪后启动，在操作面板按 n 开始本轮。
bash gear_sonic/scripts/experiments/baselinelavira.sh --task T1 --case visible_01
```

`--task` 支持 T1–T4，`--case` 使用 `cases_main.yaml` 中的条件编号；未填写的
任务/条件会列入缺项。`--config` 指向另一份实验 YAML，`--output` 指定结果根目录。
每次实际启动的完整配置和日志保存到
`outputs/experiments/baselinelavira_<任务>_<条件>_<会话编号>/`，
模型元数据记录 iPlanner 权重；`full_vln` 的记录仍独立保存。

实验脚本的 iPlanner 权重和控制参数在 `gear_sonic/config/baselinelavira.yaml`
中配置，例如将 `components.iplanner.checkpoint` 改成已有权重的绝对路径。
缺少源码、提示词、权重或模型配置时，`--dry-run` 会显示 `ready: false` 和具体路径。

独立调试入口 `baselinelavira.py` 也保留，其 `--config` 接收完整 runtime 配置：

```bash
# 生成配置并检查服务命令、源代码和权重路径，不启动机器人或模型服务。
python baselinelavira.py --dry-run

# 使用已安装的权重启动；其他操作键与原 launcher 相同。
python baselinelavira.py
```

默认权重位置是 LaViRA iPlanner 目录下的 `checkpoints/iplanner.pth`。
2026-09-13 已下载并安装 [iPlanner 官方 README](https://github.com/leggedrobotics/iPlanner)
链接的 [plannernet.pt](https://drive.google.com/file/d/1UD11sSlOZlZhzij2gG_OmxbBN4WxVsO_/view)，
约 213 MB、53,326,864 个参数。官方文件保存的是完整网络对象，已通过限定类的
`weights_only=True` 加载，无损转换为 LaViRA 需要的 `(CPU state_dict, metric)`；
与原版 `PlannerNet(encoder_channel=16)` 严格匹配，每个张量都验证相等，没有重训或修改参数。
原文件保留为 `checkpoints/plannernet.official.pt`，下载来源、两份文件的 SHA-256
及转换信息保存在同目录 `provenance.json`。官方说明此公开权重未经实机数据适配；
LaViRA G1 的 README 没有给出可核实的专用权重版本，因此不把它称为作者实机实验的同一份权重。

服务使用已有的 `navdp` conda 环境，依赖 torch、torchvision、Flask、
OpenCV、Pillow、NumPy 和 PyYAML。可通过 `--conda-env`、`--device cpu`、
`--iplanner-root`、`--model-config` 覆盖模型运行参数；`--config` 可选择
另一份完整 Groot runtime 配置。原 launcher 的参数继续透传。

生成的完整配置保存在 `outputs/baselinelavira/runtime_*.yaml`，所有进程
读取同一份快照，退出 tmux 后仍保留。相机服务器的启动方式沿用现有流程。
默认启动入口 `gear_sonic/scripts/launch_inference.py` 继续使用 X-NavDP。

已经在 RTX 3090 / PyTorch 2.5.1+cu124 上，使用现有实验的胸部 RGB-D 快照，
通过原版服务的临时本机 HTTP 端口验证模型加载、51 点轨迹输出、Groot 客户端和
Pure Pursuit。改变 RGB 不改变轨迹，改变深度或目标点会改变轨迹。
检查脚本还需要 `pyzmq`、`msgpack`、`msgpack-numpy` 来导入 Groot 公共接口；
这三个依赖已补齐。可复验：

```bash
conda run --no-capture-output -n navdp python -m gear_sonic.utils.inference.iplanner.check \
  --output outputs/baselinelavira/check_cuda.json
```

默认使用合成 RGB-D，可加 `--snapshot <已有的 snapshots/.../vla_*/ 目录>` 使用
已保存的真实数据，或加 `--device cpu` 检查 CPU 推理。此检查只启动临时
`127.0.0.1` 模型服务并在完成后关闭，不连接相机或机器人，不发出运动指令。
实机导航效果仍需实际试验验证。

网络本身只输入米制深度和目标点 `[x, y, z]`；当前 HTTP 服务把 `z` 固定为 0，
RGB 仅解码、未送入网络。`/navigator_reset` 接收相机 3×3 内参、`batch_size=1`
和 `stop_threshold=0.1`，后两者在原服务中未实际使用，内参只保存到 agent，
不参与网络前向。服务不要求地图、点云、机器人速度、历史图像或里程计。
模型配置负责深度缩放后的输入大小 360×640、深度阈值和传感器偏移；
`max_goal_distance` 虽从 YAML 读取，但此版真正执行的是服务端每轴 ±5 m 裁剪。

完整 Groot baseline 仍需要胸部相机内参及正确的深度尺度来把 VA 像素投影为
机体坐标目标（x 前、y 左，单位米），并依赖带时间戳的 FAST-LIO 位姿进行目标
更新及轨迹坐标转换。现有转向朝向反馈、雷达保护和 Sonic 执行器仍照常运行；
这些是外层导航执行链的依赖，不是新增的神经网络输入。

iPlanner 接口接收现有 SensorGateway 的 BGR 图像和米制深度，仅在 HTTP
编码层转换为 PNG 和 0.1 mm 深度单位。uint16 最大可表示 6.5535 m，
更远的有效深度在传输时饱和到该值，避免整数溢出；无效深度置零。
全零、全非有限或全非正深度在请求模型前拒绝，因为原网络即使收到全零深度
也会输出轨迹；失败交由现有导航推理失败及轨迹过期流程处理。
目标按原服务限制裁剪到每轴 `[-5, 5]` m；返回轨迹保留 x 向前、y 向左，
使用当前图像对应的 FAST-LIO 位姿补偿机器人在重规划之间的运动。
iPlanner 的 fear 分数与 LaViRA G1 原执行循环一样不作为额外停止阈值。
非有限、少于两个点或零长度的轨迹按现有推理失败流程处理，不能作为到达证据。

选点采用 VA 所见 RGB 对应的深度帧，从 bbox 底边中心周围 7×7 窗口取
有效深度的第 30 百分位，投影为机体坐标；保留当前采集流程，但不再使用
原来的框中心和多帧距离平均。置信度不足、无效深度仍停止本次目标生成，
不会退回无依据的向前移动。

轨迹控制使用 LaViRA 的 Pure Pursuit，导航期间不创建或调用 MPC：

- 前视点：从当前索引向前找第一个距离超过 0.5 m 的轨迹点。
- 速度：按当前设置，正常前进范围 0.3–0.4 m/s，直行 0.4 m/s，随转弯角度
  减速但不低于 0.3 m/s；到达目标、无有效轨迹或触发保护仍输出零速。
  对应 `target_speed: 0.4`、`min_forward_speed: 0.3`、`max_forward_speed: 0.4`。
  角速度上限 0.5 rad/s。在到达容差外仍根据轨迹修正横向偏差，
  进入 2.0 m 容差后停止；控制器不再使用原版 G1 固定的近距离停止转向分支。
- 到达判定和重规划始终使用 VA 指定的固定世界目标，模型预测的短轨迹或弯曲轨迹
  不能把目标替换成自己的末端。这是相对原版 G1 执行器的显式调整，防止尚未接近
  VA 目标便误报到达。默认不截短首条轨迹（`safe_distance: 0.0`）；若配置安全
  截短距离，仅截短首条控制轨迹，不移动 VA 目标，截短后无有效轨迹按推理失败处理。
- 重规划间隔 0.1 s，目标容差 2.0 m，单段导航最长 60 s。

上述值集中在 `components.iplanner_navigation`；旧 `navdp.goal_tolerance_m`
和 `xnavdp_mpc` 不控制 baseline 的到达判定及速度生成。导航消息、里程计
新鲜度检查、急停、任务代次隔离及 20 Hz 速度发布仍由现有 Groot 执行链负责。

为保留现有相机和对齐阶段，原版导航提示词的方向集合适配为当前五个视角，
不允许未采集的后方视角。VA 的 STOP 带有效 bbox 时继续最终接近；进入
ALIGN 仍需当前 Groot 的双视角全局目标交接验证。该验证仅判断阶段交接，
不参与选择导航 waypoint。导航历史沿用当前相机结果图，ALIGN/MANIPULATE
继续使用现有 LA/VA 与 BasePose 流程。

`xnavdp_http`、`navdp_velocity`、日志组件 `navdp` 和导航消息中的来源名
保留为兼容接口名称，实际后端由 `navigation_policy.backend` 指定，
tmux 模型服务窗格及服务日志会显示 iPlanner。服务仍由原 launcher 管理，
不需要另开原 LaViRA 的机器人驱动或控制程序。

模型、服务和导航提示词直接从本地 Uni-LaViRA 目录读取；轨迹控制公式和
目标投影方法适配在 `gear_sonic/utils/inference/iplanner/`。
作者及原始许可见该源码目录及 Groot 的第三方声明。

2026-09-13 对照了官方 [Uni-LaViRA G1 提示词](https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code/blob/main/real-world-code/unitree_g1/prompts.py)
及 [控制器](https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code/blob/main/real-world-code/unitree_g1/robot/robot_controller.py)，
并检查原目录中 `baselinelavira_T1_visible_02_b58fccf87853` 的实际请求和回复。
g3 的第二步 LA 在进展分析中写 Image 5 / 左前方 45°，却返回 `left`（90°）；
转向后的 VA 自述蓝篮桌不可见，仍选了无关电脑桌。前者是模型输出自相矛盾及
校验缺失，后者同时受到旧 prompt 缺少目标不可见分支、转向前后方向描述混用的影响。
目前没有确认基础世界/机体旋转矩阵的左右符号写反；离线左右对称及多朝向测试
用于检查这一点，不能替代实机验证，也不能证明所有偏左现象已经解决。

最小实验日志现在保留 VA 框/深度投影、全景完成、最终朝向，以及
`IPLANNER_PLAN_ACCEPTED` / `IPLANNER_CONTROL`。后两类每秒最多各一条，
包含规划位姿、请求/执行世界目标、局部/世界轨迹、当前前视点与速度，
以及目标距离、配置的到达容差、轨迹末端距离。同一条控制记录还保存现有 SONIC
遥测中的实测 yaw、相对初始朝向、朝向设定值及数据年龄，用于与 FAST-LIO
朝向对照；缺少遥测时记为 null。这些记录用于追踪偏差从哪一环出现，
不改变朝向控制。此前缺少这些记录，无法从几个阶段端点恢复真实连续轨迹。
