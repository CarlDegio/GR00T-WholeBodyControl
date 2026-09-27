# iPlanner baseline

从本机 `GR00T-WholeBodyControl-baselinelavira` 的当前工作目录迁入导航适配和启动入口。
本分支直接使用当前共享的 LA/VA、FlexibleMissionAgent、实验记录、BasePose、VLA 后端切换
及 PLANNER 交接实现。没有复制源分支较旧的 Agent 或现场任务/相机配置。

```text
同一个 LA 选择目标 → 同一个 VA 和 RGB-D 目标投影
  → iPlanner 轨迹 → Pure Pursuit → 现有速度执行器 → SONIC
```

完整对齐、操作完成检查、多次操作顺序、每个 VLA 的固定训练 prompt，以及任务取消和
整轮时限都沿用当前 `full_vln`。所有配置先读取当前 `launch_inference.yaml`，再通过实验
YAML 的 `runtime_overlays` 应用 `baselinelavira.yaml`，最后加载所选 mission。

## 启动

在项目根目录运行：

```bash
# 检查配置和本地模型文件，不启动任何服务或创建实验目录。
bash gear_sonic/scripts/experiments/baselinelavira.sh --mission gear_sonic/config/missions/bag_then_bin.yaml --dry-run

# 使用整段任务 prompt，由 LA 自由安排导航与多次 VLA。
bash gear_sonic/scripts/experiments/baselinelavira.sh --mission gear_sonic/config/missions/bag_then_bin.yaml

# 原单任务实验入口仍可用。
bash gear_sonic/scripts/experiments/baselinelavira.sh --task T1 --case visible_01
```

两个 VLA 模型服务仍须预先启动；iPlanner 服务由 launcher 自动启动。在机器人和模型就绪后
按 `n` 开始整条试验。`--task/--case` 决定实验标签、布局/起点、预算和 milestones，必须
对应实际实验条件；不指定时默认 T1/visible_01。日志写入当前项目
`outputs/experiments/baselinelavira_<任务>_<条件>_<会话>/`。

## 导航差异和依赖

`gear_sonic/config/baselinelavira.yaml` 保留源分支的设置：

| 参数 | 值 |
| --- | --- |
| 导航后端 | iPlanner（替代 X-NavDP） |
| 轨迹跟踪 | Pure Pursuit（替代 MPC） |
| 到达容差 | 2.0 m |
| 前进速度范围 | 0.3–0.4 m/s |
| 最大转速 | 0.5 rad/s |
| 前视距离 | 0.5 m |
| 重规划间隔 | 0.1 s |
| 单段导航时限 | 60 s |

到达判定保持 VA 指定的世界目标，不用模型短轨迹的末端替代。过期轨迹、无效深度、
异常轨迹、里程计过期、取消及 generation 隔离沿用迁入的导航检查和共享执行链。
全景转向仍使用当前 SONIC 朝向反馈。

模型源码位于 `/home/user/Project/uni-lavira-code/real-world-code/unitree_g1/iplanner`，
权重为该目录下的 `checkpoints/iplanner.pth`，模型配置为 `configs/iplanner.yaml`。
服务默认使用 `navdp` conda 环境和 CUDA。launcher 保留当前分支的 conda Python 隔离、
GPU 环境、demo controls、分阶段启动及进程清理。

`xnavdp_http`、`navdp_velocity` 和内部 `navdp` 窗格标识继续作为兼容接口名称；
实际模型由 `components.navigation_policy.backend` 决定，窗格标题显示 iPlanner。
实验元数据保存 iPlanner checkpoint，最小日志保留每秒最多各一条的
`IPLANNER_PLAN_ACCEPTED` / `IPLANNER_CONTROL`。

独立配置辅助入口 `baselinelavira.py` 和离线模型检查工具
`gear_sonic.utils.inference.iplanner.check` 也已迁入。常规多 VLA 实验使用上面的 bash。
离线回归验证共享 Agent 行为、HTTP 编解码、目标与轨迹控制和实验入口；不代表真机效果已验证。
