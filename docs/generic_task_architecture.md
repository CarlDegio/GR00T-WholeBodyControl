# 通用任务链的增量实现

已在 `pnp_demo` 接入。配置及使用方法见 [多 Skill 任务指南](multi_skill_missions.md)，完整 overlay 见 [interleaved_skills.yaml](../gear_sonic/config/missions/interleaved_skills.yaml)。

任务顶层由导航和操作组成，可以穿插，也可以连续操作。每次操作在代码中强制执行自己的 Align，再启动一次 VLA；相同 Skill 或相同目标也不能复用之前的对齐结果。

```text
[ALIGN A → SKILL A] → NAV B → [ALIGN B → SKILL B]
                          → [ALIGN C → SKILL C] → NAV D
```

沿用原有 MissionAgent / LaViRAAgent、服务进程、PolicyRouter 和控制入口。没有新增 MotionBridge、统一控制所有者、统一保持开关或抓取检测。

| 修改位置 | 职责 |
| --- | --- |
| `lavira/mission.py` | 解析逻辑 Skill 定义及 navigate/skill 步骤，顺序执行，消费子步骤结果；失败或取消即结束任务 |
| `lavira/agent.py` 的 `run_navigation()` / `run_operation()` | 导航只规划当前目标；操作独立执行 Align → VLA → PLANNER 交接 |
| `agent.py` 的导航 prompt 和 validator | 显式导航只允许 MOVE_TO 或报告失败，Align 和操作顺序由代码安排；保留旧入口的 LA 契约 |
| `agent.py` 的操作上下文和门控 | 记录当前 operation_id、对齐动作/段编号、位置、朝向和时间；每次启动 VLA 前重新验证 |
| `lavira/service.py` | 传入 Skill 注册表；接收单 Skill 人工完成事件；仅整条任务的 worker 结果进入原终止路径 |
| `runtime/gateway/control.py` / `services/control.py` | 校验本次 BasePose 成功结果与 VLA 操作编号的对应关系；过滤过期/重复完成；保持原 PLANNER 交接确认 |
| `vla/runtime.py` | 在成功出口把 hold/default 映射到现有上半身状态和左右夹爪目标 |
| `utils/operator/console.py` | 增加 `:skill_done` 文本命令提示，保留原单键控制 |

每个步骤清空感知历史、TODO、导航就绪标志及旧对齐状态，保留同一 generation 下递增的 skill_id / segment_id / scan_id。操作入口用当前实测朝向建立参考并拍摄新 RGB-D 图像，不回转到旧导航环视的朝向。对齐识别使用本次 align 描述；VLA 始终使用该 Skill 自己的训练提示。

Align 不要求之前存在 NAV 历史，但必须满足当前可见目标和深度范围条件，随后通过既有 BasePose 对齐及 VA 共视检查。目标不在可操作范围时失败，需要在配置中先安排导航。对齐失败可按 `align.max_attempts` 重新尝试；每次重新拍摄和执行。VLA 入口再次检查当前操作、编号、取消状态、位移、朝向、对齐时间和原 readiness。网关中的 BasePose 对齐结果只能消费一次；导航、新对齐和取消均清除旧结果。

操作完成方式支持 `visual` 和 `manual`。视觉完成沿用现有 VA；人工完成通过 `:skill_done` 加回车，只结束当前等待人工确认的 Skill。G/H 仍标记整条任务成功/失败。异步失败和超时不能当作完成，重复完成不推进两次。没有根据名字推断抓取/投放，也没有增加后端未提供的 done 信号。

上半身与夹爪只在上层任务出口成组选目标，底层仍保留原有独立状态和控制逻辑：

| 上层出口 | 上半身状态 | 夹爪状态 |
| --- | --- | --- |
| `exit: hold` | 选择现有 hold_current 路径，在进入 PLANNER 保持时由原 C++ 逻辑采样实测角度 | 沿用当前左右手动开合目标 |
| `exit: default` | 选择默认 planner 手臂两侧姿态，走原平滑过渡 | 将左右手动目标设为打开 |

新操作的出口只有这两种，不能单配上半身或夹爪出口。旧配置没有 exit 时仍走旧行为，不会因为 navigation_upper_body 为 default 就额外打开夹爪。取消或失败不执行成功出口的 default 赋值。

这层赋值不改变 j / [ / ]、手动夹爪优先级、PLANNER / POSE 模式切换、夹爪发布循环或原 3 秒 POSE 初始姿态过渡。下一次 VLA 仍调用原入口，不自动获得夹爪控制权。导航和对齐不会反复覆盖用户后来修改的手动目标；不新增底层统一锁，也不承诺上半身和夹爪在同一物理时刻到位。

完成确认后沿用 completion_settle_seconds 和 PlannerHandoff 时序，再推进下一步。hold 采样发生在 PLANNER 保持入口，而非 VA 或人工完成事件到达的瞬间。携带姿态可以沿用到后续导航/对齐，但下一 Skill 仍执行原 POSE 初始姿势过渡；本轮未改为任意终态直接衔接下一策略。

后端与 Skill 分开配置：多个逻辑 Skill 可映射到同一 PolicyServer。仍只有一个 VLA 服务、发布者和推理线程；多个后端串行预热、串行请求。现有单任务及 ExperimentAgent 入口兼容，单试验 overlay 与 mission_steps 不混用。

回归位于 `gear_sonic/tests/test_multi_skill_mission.py` 及既有 LaViRA、网关、VLA、手动夹爪、POSE 过渡测试中。mock 使用真实编排和状态机，模拟传感器、模型与 C++ 状态；不启动机器人。
