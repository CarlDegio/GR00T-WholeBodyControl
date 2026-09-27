# 多 Skill 与连续导航

需要从一整段自然语言指令自由安排导航和操作时，使用
[`bag_then_bin.yaml`](../gear_sonic/config/missions/bag_then_bin.yaml) 的 `flexible_mission: true`。
`mission` 是唯一的整段任务输入，`skills` 只是能力与模型注册表，不是执行顺序；不配置 `mission_steps`。
复用同一个 LA 客户端、VA 客户端和原 `LaViRAAgent.run()` 动作循环，没有额外规划层或拆分子任务调用。
LA 每步读取完整指令、完整 TODO 和当前视觉，自主选择 MOVE_TO、ALIGN、MANIPULATE 或最终 COMPLETE。
导航目标可以随任务进展更新；导航可出现在操作前、操作之间或操作之后，也支持先操作及连续操作。
每次 VLA 执行前仍必须新鲜对齐；直接操作时保留现有局部可见性、深度和 BasePose/VA 就绪检查。

实验对比入口现在也可使用这套自由顺序流程：给 `full_vln.sh`、`nav_direct_vla.sh`
或 `geometric_vla.sh` 添加 `--mission gear_sonic/config/missions/bag_then_bin.yaml`。
它们分别在每次操作前使用完整对齐、跳过对齐、简单几何对齐，导航和多 VLA 调度仍由同一个 LA
完成。几何分支的目标配置在每个 Skill 的 `geometry_target`，其余两种方法不使用此字段。
`baselinelavira.sh` 也支持相同参数，使用完整对齐，将导航换成 iPlanner + Pure Pursuit。
`unilm_nav.sh` 同样支持；每次对齐改用对应技能的 `vla_prompt` 和胸部 RGB-D 的
UniLM VLM 位姿推理，详见 [UniLM 实验入口](unilm_nav.md)。
完整命令、日志和实验标签说明见[实验入口](../gear_sonic/scripts/experiments/README.md#自由顺序多次-vla)。

LA 模型和 VA 接口保持原配置。自由顺序模式只扩展 LA 动作约定：ALIGN 携带逻辑 `skill_key` 和
从整段指令提取的本次 `task`，MANIPULATE 必须选择相同 Skill；操作结束后继续同一循环，
由 LA 在完整 TODO 全部处理后返回 COMPLETE。`task` 保留 LA 对整段任务的理解，不能覆盖配置的操作描述。
LA 每次都能看到所有技能的 `vla_prompt`（请求中作为 `description`），据此选择 `skill_key`。
VA 使用所选技能配置中的 `vla_prompt` 和实时图像推理对齐对象、朝向参考，并沿用原 POSTCHECK。
`vla_trained_prompt` 只由运行时固定传给对应模型，
不暴露给 LA，也不接受 LA 改写。两者不再承担两份单独任务的作用。

下面的 `navigate/skill` 列表是仍兼容的显式编排方式；本次整段 prompt 模式不使用它。

`components.lavira.mission_steps` 按顺序执行导航和操作。`type: navigate` 只完成当前导航；`type: skill` 必须执行本次 Align，再运行本次 VLA，交接回 PLANNER 后进入下一步。支持操作后继续导航、连续操作和重复使用同一个 Skill。失败或取消结束整条任务。

示例 [interleaved_skills.yaml](../gear_sonic/config/missions/interleaved_skills.yaml) 包含三次操作和两段导航，使用两个后端：

```text
[ALIGN A → SKILL A / hold] → NAV B → [ALIGN B → SKILL B / hold]
                                  → [ALIGN C → SKILL C / default] → NAV 门口
```

第一步示范已经站在目标附近时直接操作，仍执行对齐。现场若需要先靠近，须在它前面添加 navigate。示例中的目标、训练提示和完成条件应按实际任务修改；代码不检测是否抓住物体，也不推断是否需要携带。

## 配置

`components.vla.backends` 注册已经运行的模型服务；`components.lavira.skills` 定义逻辑 Skill。多个 Skill 可选择同一后端。
新配置使用两个独立提示词：`vla_prompt` 给 Agent/VA 描述现场任务，`vla_trained_prompt` 是实际发给模型的训练原文。
后端 `prompt` 用于预热和默认手动 POSE，应通过 YAML anchor 引用同一训练原文。
旧 `prompt` 写法仍兼容，省略 `vla_trained_prompt` 时才沿用原来 Agent/VLA 共用提示词的行为。

```yaml
components:
  deploy:
    demo_controls: true
  vla:
    default_skill: model_a
    prewarm_backends: true
    backends:
      model_a: {host: 127.0.0.1, port: 29999, prompt: collect the pill bottle}
      model_b: {host: 127.0.0.1, port: 30000, prompt: collect the rubbish}
  lavira:
    skills:
      action_a:
        backend: model_a
        prompt: collect the pill bottle
        completion: {kind: manual}
        completion_settle_seconds: 0
        exit: hold
      action_b:
        backend: model_b
        prompt: collect the rubbish
        completion: {kind: visual}
        expected_postcondition: The paper balls are inside the basket.
        completion_settle_seconds: 0
        exit: default
    mission_steps:
      - {id: approach, type: navigate, target: the table with the pill bottle}
      - id: a
        type: skill
        skill: action_a
        align: {target: the blue basket next to the pill bottle, max_attempts: 3}
      - {id: move_b, type: navigate, target: the table with paper balls}
      - id: b
        type: skill
        skill: action_b
        align: {target: the blue basket next to the paper balls}
      - {id: return, type: navigate, target: the door}
```

| 字段 | 含义 |
| --- | --- |
| `id` | 本次步骤的唯一名称，同一 Skill 重复调用需要不同 id |
| `type` | navigate 或 skill；Align 不单列为顶层步骤 |
| `target` | navigate 的导航目标；可另写 mission，省略时自动生成导航指令 |
| `navigation_mode` | 导航可选 object_nav（默认）或 vln |
| `skill` | `lavira.skills` 中的逻辑 Skill 名，选择该定义的 backend 和提示词 |
| `vla_prompt` | 给 Agent/VA 的任务描述；新格式必须与 vla_trained_prompt 一起填写，不能与旧 prompt 同时使用 |
| `vla_trained_prompt` | 实际传给 VLA 的语言输入；保留大小写、标点和尾部空格，切换步骤时独立设置 |
| `align.target` | 每次操作必填的对齐目标描述，不能复用上一次 Align |
| `align.yaw_align_target` | 可选的朝向参考物描述；省略时沿用 VA 识别 |
| `align.max_attempts` | 默认 3，范围 1–20，也受 lavira.max_steps 的预算限制 |
| `completion.kind` | 显式步骤可选 visual/manual；自由顺序模式使用 visual，并可省略 expected_postcondition，由 LA 给出本次操作的完成条件 |
| `completion_settle_seconds` | 确认完成后继续执行 VLA 的时间，默认 5 秒，允许 0–30；示例显式设为 0 |
| `exit` | 每个 Skill 定义必填 hold/default，某次操作可覆盖为另一种选择 |
| `timeout_seconds` | 可选，仅显式 visual Skill；从 VLA 确认 active 后计时，覆盖本次操作的全局超时，不包含导航、对齐和启动过渡 |
| `on_timeout` | 默认 fail；设为 continue 时必须同时配置 timeout_seconds，到时停止操作并交接至下一步，不表示物理成功 |

Skill 步骤也可直接写 `skill_key`（后端名）、`vla_prompt`、`vla_trained_prompt`、完成配置与 exit，省略逻辑注册表；旧 `manipulation_prompt` 仍兼容。使用逻辑 Skill 引用时，backend 和提示词放在定义中，不能在步骤中覆盖。未知后端、缺失 Align、拆分上半身/夹爪出口等配置在启动前报错。

操作入口建立当前朝向参考并获取新图像；对齐目标须可见且位于现有 NAV 交接深度范围（默认 0.3–3.0 米）。之后复用 BasePose 控制和 VA 对齐就绪检查。越界或无有效深度时不会启动 VLA，需要配置先导航。对齐结果绑定当前操作、generation、对齐动作和段编号；启动 VLA 前还检查位移、朝向与时效。

## 完成与出口

`visual` 沿用 VA 完成检查。`manual` 等待人工确认：看到 `SKILL_AWAITING_OPERATOR` 后，在控制台输入 **`:skill_done` 再按回车**。冒号进入命令输入模式；该命令只结束当前等待确认的 Skill。原 `G/H` 仍表示整条任务成功/失败，不用于推进单个 Skill。人工完成同样受 manipulation_timeout_seconds 超时限制（默认 180 秒）。

| 出口 | 上半身 | 夹爪 |
| --- | --- | --- |
| `hold` | 使用原 hold_current，在 PLANNER 保持入口采样实测 17 个关节角度 | 保留当前左右手动开合目标 |
| `default` | 恢复 planner 默认手臂两侧姿势，走原平滑过渡 | 左右手动目标都设为打开 |

上层只选择这两种组合，沿用既有独立状态和控制接口。原 `j` 同时切换双夹爪、`[` / `]` 分别控制左右夹爪，默认打开，手动目标优先于 VLA 输出；这些行为没有改成新的统一锁。任务出口只赋值一次，后续导航/对齐不会反复覆盖用户的新手动指令。失败和取消不执行成功出口的恢复默认。

交接顺序：完成确认 → 配置的延迟 → 清空动作缓存并使旧推理失效 → 发出带唯一编号的 PLANNER 指令 → C++ 返回同编号、PLANNER 模式与就绪标志 → 连续确认 2 秒 → 下一步。C++ 在 10 秒内没有确认则失败；旧二进制缺少确认字段时也不放行。重复完成不延长延迟，取消使旧响应失效。

hold 保持的是原 PLANNER 入口采样到的上半身及当前手动夹爪目标，不是 VLA 完成瞬间的动作快照。进入下一次 POSE Skill 仍有原 3 秒初始姿态过渡；本轮没有改变这一控制逻辑，也未自动把夹爪控制权交给 VLA。

### 抓袋子后导航至垃圾桶

对齐结果有效期由 `components.lavira.operation_alignment_max_age_seconds` 控制，默认
30 秒，用于容纳对齐后 LA 的决策延迟；它独立于 `vla_start_timeout_seconds`（只等待
VLA 服务启动确认）。位移超过 0.1 米或朝向偏移超过现有阈值时仍拒绝启动操作；
`OPERATION_ALIGNMENT_INVALIDATED` 会记录实际时间、位移和朝向差，便于定位失败原因。

[bag_then_bin.yaml](../gear_sonic/config/missions/bag_then_bin.yaml) 在 `components.lavira.mission`
提供整段导航＋操作示例。修改这一段即可改变路线和顺序，不需要写各段导航目标或操作列表。
抓袋子的 Skill 由 VA 检查完成，确认后不额外执行 VLA；若执行满 40 秒仍未确认，则以
`timeout_continue` 停止 VLA，保持上半身及当前手动夹爪目标，等待原 PLANNER 交接确认后交还 LA 规划下一步。
超时不等待正在进行的云端 VA 请求；过期回复不会发布判定或控制下一步。此计时是软件调度时间限制，
不是硬实时保证。踩桶 Skill 同样由 VA 判断完成，沿用全局 180 秒操作超时，超时终止任务。
两个模型服务需提前启动：`bag_vla` 默认连接 `127.0.0.1:29999`，`bin_vla` 默认连接
`127.0.0.1:30000`，地址和端口可分别修改。

当前权重和训练提示词如下（拿袋于 2026-09-23 核对，踩桶于 2026-09-25 切换到新权重）：

| checkpoint（位于 openpi_sonic/checkpoints） | 数据集 | vla_trained_prompt |
| --- | --- | --- |
| pi05_g1_sonic_full_collect_bag/pi05_g1_sonic_full_collect_bag_77/29999 | CollectBag | `collect the pill bottle` |
| pi05_g1_sonic_full_throw_bag_big/pi05_g1_sonic_full_collect_bag_big_187/29999 | ThrowBagBig | `collect the pill bottle` |

两个模型均须使用训练原文 `collect the pill bottle`。新踩桶模型训练运行 `uoxpkyg4` 启用
`prompt_from_task=True`，无提示词替换；188 个 parquet 的 215,896 帧全部为 `task_index=0`，
实际任务映射为这句话。部分 episode 描述虽然写着扔袋子，训练读取的是帧级任务索引。
核验记录：`/tmp/openpi-sync-20260925/verification.json`。LaViRA/VA 的操作描述仍保持踩桶扔袋子的真实语义。

踩桶 PolicyServer 使用下面的配置和权重，端口保持 30000。新模型使用头部、左右腕三路图像；
现有客户端仍传四路图像，服务端按训练配置选用三路。先结束占用该端口的旧模型服务，再执行：

```bash
cd /home/user/Project/openpi_sonic
OPENPI_DATA_HOME=/home/user/Project/openpi_base \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_g1_sonic_zmq_policy.py \
  --host 127.0.0.1 \
  --port 30000 \
  --policy.config pi05_g1_sonic_full_throw_bag_big \
  --policy.dir checkpoints/pi05_g1_sonic_full_throw_bag_big/pi05_g1_sonic_full_collect_bag_big_187/29999 \
  --default-prompt "collect the pill bottle" \
  --verbose-timing
```

修改整段任务描述或端口后，重新生成完整配置：

```bash
.venv_teleop/bin/python -m gear_sonic.scripts.prepare_mission --overlay gear_sonic/config/missions/bag_then_bin.yaml --output outputs/missions/bag_then_bin.yaml
```

自由顺序模式的操作超时继续会记录 `SKILL_TIMEOUT_CONTINUING` 和 `MISSION_OPERATION_TIMEOUT_CONTINUED`，
操作历史 VA 结果为 `UNKNOWN`，整条流程的结果原因是 `mission_completed_with_timeouts`。
这表示顺序执行结束，不表示所有动作成功，也不会补写人工成功或物理完成时间。
安全故障、取消、连续 VA UNKNOWN、请求错误及窗口数耗尽仍会终止任务；对齐目标不满足距离条件时
交还 LA 选择导航，普通对齐不就绪也由 LA 根据 VA 结果重试或导航。
`on_timeout: continue` 只允许时间预算耗尽后继续。未启用 flexible_mission 的单任务与显式编排保持原行为。

## 后端与启动

### 真机 smoke：不启动 VLA 后端，假设操作成功

在任务 overlay 后叠加 `smoke_assume_vla_success.yaml`。真实 LA、VA、相机、导航和
BasePose 对齐照常运行；每次真实对齐通过后跳过 VLA 操作，LA 根据整段任务继续规划。
该模式不会进入 POSE，也不验证抓袋子、踩桶或投袋子的物理效果。首个 VLA 的 40 秒
超时配置保留，但 smoke 跳过操作，因此不等待这 40 秒。

```bash
cd /home/user/Project/GR00T-WholeBodyControl
.venv_teleop/bin/python -m gear_sonic.scripts.prepare_mission \
  --overlay gear_sonic/config/missions/bag_then_bin.yaml \
  --overlay gear_sonic/config/missions/smoke_assume_vla_success.yaml \
  --output outputs/missions/bag_then_bin_smoke.yaml
.venv_inference/bin/python gear_sonic/scripts/launch_inference.py \
  --config outputs/missions/bag_then_bin_smoke.yaml
```

无需启动两个 PolicyServer。保留 launcher 启动的本地 VLA 服务进程，它仍负责 PLANNER
消息转发；该进程不连接、ping 或预热任何 VLA 后端，手动 POSE/恢复策略也被禁用。
看到 `SMOKE_VLA_BACKENDS_DISABLED`，并按原流程使机器人就绪后按 `N`。
不用等待 `VLA_BACKENDS_WARMED`。空格取消仍有效。

每次操作记录 `SMOKE_VLA_ASSUMED_SUCCESS`，TODO 明确标注 `SMOKE`，VA 操作结果为
`UNKNOWN`；最终原因是 `smoke_mission_completed`，不能作为真实操作成功。该模式
不允许与正式单试验 experiment overlay 混用。恢复真实操作时去掉 smoke overlay，
重新生成普通配置并启动两个 VLA 后端。

### 正常操作

后端配置可选 timeout_ms（默认 15000），default_skill 用于手动 POSE 的默认模型。服务端需自行启动，任务代码不负责加载检查点或启动 PolicyServer。

只有一个 VLA 服务、动作发布者和推理线程。各后端的 ping、预热和推理共用单请求通道。SensorGateway 有新鲜四视角图像与机器人状态后，各模型串行执行一次推理并丢弃结果。看到 `VLA_BACKENDS_WARMED` 后再开始任务；预热未完成时拒绝进入 VLA POSE。预热失败可检查 `VLA_PREWARM_FAILED` 日志。

先生成 launcher 接受的完整配置。该命令只写 YAML，不启动服务：

```bash
.venv_teleop/bin/python -m gear_sonic.scripts.prepare_mission --overlay gear_sonic/config/missions/interleaved_skills.yaml --output outputs/missions/interleaved_skills.yaml
```

操作者手动启动时使用本仓库路径（下条命令会启动真机 inference）：

```bash
.venv_inference/bin/python gear_sonic/scripts/launch_inference.py --config outputs/missions/interleaved_skills.yaml
```

## 兼容与验证

姿态覆盖、跨模式手动夹爪锁定和 C++ 交接确认限定在 demo 中使用：`components.deploy.demo_controls: true` 让 inference launcher 传入 `--demo-controls`。仓库的 demo 主配置和两个任务示例均已显式开启。直接运行 `deploy.sh` 时，这些扩展默认关闭；手动启动 demo 需添加此参数。

`launch_data_collection.py` 不读取 inference 的 deploy 设置，也不传入该参数。采集保持上游 `agent_full` 的 PICO 姿态、冻结上半身目标、手爪控制及状态报文；demo 扩展字段在采集进程中按未知字段忽略。采集入口、采集配置、PICO 和导出代码保持不变。开关在 C++ 进程启动时确定；从 demo 转为采集须通过采集入口启动独立 C++ 进程，不能复用带 `--demo-controls` 的进程。

旧 [two_skills.yaml](../gear_sonic/config/missions/two_skills.yaml) 仍可用：未写 type 的步骤保持原 mission/global_target → 自动导航/对齐/操作流程，省略 skill_key 表示只导航。旧 navigation_upper_body 支持 default/hold_current；没有新 exit 字段时保留原夹爪状态，不会因上半身恢复默认而额外打开夹爪。空 mission_steps 保留单任务和实验入口。默认运行配置未启用示例任务。

任务内部共用一个 generation，动作、段和扫描编号递增；切换步骤清空视觉上下文与旧对齐状态。只有整条列表结束才发布最终任务结果。mission_steps 不能与 components.experiment 单试验配置混用，启动时提前检查。

无需机器人、GPU 或云模型的 mock 回归：

```bash
.venv_teleop/bin/python -m pytest -q gear_sonic/tests/test_multi_skill_mission.py gear_sonic/tests/test_vla_backends.py gear_sonic/tests/test_manual_grippers.py gear_sonic/tests/test_pose_entry_transition.py
```

测试使用真实任务编排、控制网关和 VLA 状态机，模拟相机、模型与 C++ 状态。覆盖多 Skill 穿插导航、每次重新对齐、人工/视觉完成、出口赋值、旧控制兼容、重复通知、过期对齐、取消及交接超时。实际动作效果仍需操作者验证。
