# T1–T4 mission 配置

四个任务共用 T4 的 flexible mission 框架：整段 `mission`、LA 动态 To-do list、按需导航和重新推理、VA 对齐与完成检查、命名 VLA 后端。技能表不是执行顺序；一个技能内部可以包含多个操作动作。

所有任务的 YOLOE 对齐默认直接使用逐帧检测框和分割 mask（`raw_yoloe_tracking: false`），不经过 BoT-SORT，不要求跟踪 ID。每个角色在当前帧按检测置信度选目标；头胸切换、深度和几何检查继续生效。未配置该字段的历史 profile 也默认关闭跟踪。无跟踪时逐帧日志的 ID 为 `null`，侧移脉冲仍记录时长和控制量，但无法确认跨帧物体身份的视觉位移标为缺失。已运行进程需重启后加载。

| 任务 | YAML | 操作技能 | 所需 PolicyServer |
|---|---|---|---|
| T1 药瓶入篮 | [t1_bottle_to_basket.yaml](t1_bottle_to_basket.yaml) | 药瓶放入蓝篮，1 个 | 29999 |
| T2 袋入箱并搬箱 | [t2_bag_and_box.yaml](t2_bag_and_box.yaml) | 装袋、搬箱、放箱，合为 1 个 | 29999 |
| T3 踩桶倒垃圾 | [t3_bowl_to_bin.yaml](t3_bowl_to_bin.yaml) | 拿碗、踩桶、倒垃圾并放回碗，合为 1 个 | 29999 |
| T4 提袋导航踩桶投放 | [bag_then_bin.yaml](bag_then_bin.yaml) | 提袋、踩桶投袋，2 个 | 29999 和 30000 |

`components.mission_metadata.task_id` 决定实验任务编号，优先于命令行 `--task`；日志目录、冻结配置和统计均使用该编号。T1–T3 分别启动对应模型到 29999，不代表三个任务使用同一权重。

```bash
bash gear_sonic/scripts/experiments/full_vln.sh --mission gear_sonic/config/missions/t1_bottle_to_basket.yaml --demo
bash gear_sonic/scripts/experiments/full_vln.sh --mission gear_sonic/config/missions/t2_bag_and_box.yaml --demo
bash gear_sonic/scripts/experiments/full_vln.sh --mission gear_sonic/config/missions/t3_bowl_to_bin.yaml --demo
bash gear_sonic/scripts/experiments/full_vln.sh --mission gear_sonic/config/missions/bag_then_bin.yaml --demo
```

加 `--dry-run` 仅校验；不加 `--demo` 不开启 demo 录制。`--case` 按实际实验条件填写，未指定时沿用入口默认条件。其他支持 `--mission` 的实验方法也可使用这些 YAML。

Full VLN 的 operator 控制台中，`n` 开始完整任务；T1–T3 可按 `m` 仅执行当前站位的双相机对齐和一次完整操作。`--demo` 同样保存近场头胸视频及 mask、线。T4 下 `m` 对齐并拿袋子，`t` 对齐并踩桶扔袋子；两段各自结束并保存，`n` 仍执行完整任务。[近场入口与记录说明](../../scripts/experiments/README.md#full-vln-的近场入口m)。

T4 于 2026-09-25 将第二个技能 `step_bin.base_pose` 的头部目标距离固定为 **1.25 m**、胸部目标距离固定为 **0.90 m**，对应当天下午踩桶调参；头胸目标置信度均为 **0.30**，连续误差达标帧数 `raw_stable_frames` 为 **3**（其他技能默认 5 帧），之后仍执行 30 帧停车复核。距离是相机测得的目标距离。完整任务的踩桶阶段和 `t` 单段入口都使用这组值；`pick_bag` / `m` 继续继承基础配置（当前头部 1.10 m、胸部 0.90 m，目标置信度分别为 0.30 / 0.15）。每次对齐先恢复基础参数，再应用当前技能覆盖，因此换技能和手动对齐不会沿用上一次技能的设置。技能 `base_pose` 目前支持这两个距离字段、两个目标置信度字段和 `raw_stable_frames`，其他对齐参数仍继承基础配置；修改后需重新启动加载。

昨晚（2026-09-24）23:09 的启动命令就是上面的 T4 `full_vln.sh --mission .../bag_then_bin.yaml --demo`；23:31 的最后一个会话为 `full_vln_T4_visible_01_dcd23dc71905`。拿袋模型接 `127.0.0.1:29999`，踩桶投袋模型接 `127.0.0.1:30000`，需分别启动对应 PolicyServer。

T1–T3 依据截至 2026-09-24 最后有 `trial_start` 的 Full VLN 会话冻结配置迁移，来源写在各 YAML 的 `source_runtime`：

| 任务 | 源会话 | 最后一轮开始时间（北京） |
|---|---|---|
| T1 | `full_vln_T1_invisible_02_9e0b5ad8a2fd` | 2026-09-13 15:33:25 |
| T2 | `full_vln_T2_invisible_02_4bf9ed5e8da0` | 2026-09-17 15:05:02 |
| T3 | `full_vln_T3_invisible_02_55dba30d4677` | 2026-09-19 16:45:18 |

保留源导航描述、完整操作描述、实际 VLA prompt，以及对齐距离、容差和速度。T2 于 2026-09-25 按前一天最后的近场试验 `near_dual_T2_nominal_01_a6b8fda7622a` 校正：胸部目标距离为 1.15 m、切换阈值为 1.2 m，头部目标距离为 1.1 m、切换阈值为 1.5 m，头胸 YOLOE 目标置信度按用户最新要求均设为 0.10；T3 横向容差为 0.08 m；三个任务横向最大速度均为 0.4 m/s。T2 已显式固定该近场会话的相机及控制参数（头胸目标置信度使用上述用户指定值）；输出目录、demo 录制及禁用额外诊断 PNG 的策略保持当前实现。T2 所有任务语义中的 `open cardboard box` 已改成 `cardboard box`，手动对齐目标也为 `cardboard box`；自动流程的实际目标仍由 VA 按所选技能推理。其他共享服务参数继承当前基础配置。

T3 冻结配置实际发送给模型的训练文本是 `collect the pill bottle`，这里原样保留；操作语义仍是倒碗中垃圾。迁移依据是实际运行记录，不据此推定 PolicyServer 当前加载的权重。

与 T4 提袋后需要保持夹爪不同，T1–T3 的完整操作结束后使用 `exit: default`。每个技能超时 180 s，超时失败；整轮总预算 600 s。任务路线可修改 `mission`，但模型训练 prompt 应与对应权重保持一致。
