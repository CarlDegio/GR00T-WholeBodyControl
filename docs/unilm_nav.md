# UniLM-Nav 实验入口

从 `GR00T-WholeBodyControl-unilm-nav` 迁移的 **UniLM-Nav adapted to our platform**，
保留原三阶段 VLM 推理和固定目标位姿控制器，接入当前自由顺序、多次 VLA agent。
导航仍使用 NavDP，LA 仍负责选择 MOVE_TO / ALIGN / MANIPULATE；不固定操作顺序。

在仓库根目录执行：

```bash
bash gear_sonic/scripts/experiments/unilm_nav.sh --mission gear_sonic/config/missions/bag_then_bin.yaml --task T1 --case visible_01
```

加 `--dry-run` 可离线校验配置和 G1 几何资源。不传 `--mission` 使用原单任务实验配置。
模型服务提前启动，进入现有操作界面并完成机器人就绪后按 `n` 开始。

## Prompt 与图像

- 每次 ALIGN 使用所选技能 `lavira.skills.<key>.vla_prompt`，在运行时归一为
  `manipulation_prompt` 并作为 `start_base_pose.task` 传给 UniLM。选帧、交互点定位、
  位姿预测三次请求全部使用这个语义任务。单任务使用 `tasks.yaml` 中的 `vla_prompt`。
- `vla_trained_prompt` 只用于相应 VLA 模型的执行，保持训练字符串原文。
- 默认使用**胸部** `chest_view` RGB + `camera/chest_view_depth`，沿用公共 BasePose
  胸部相机外参和源帧内参，高度 1.0 m。
- 导航后使用当前操作之前已到达导航段的胸部 RGB-D/里程计历史；没有前置导航时
  重新采集五帧。每次 ALIGN 消费并清理历史，下一操作不会使用上一操作的旧帧。
  导航历史不足五帧时明确失败，保留源方法规则。

## 配置与完成条件

参数在 `gear_sonic/config/unilm_nav.yaml`：`qwen3.8-flash`、关闭 thinking、2 Hz 采样，
三阶段正常各调用一次，格式错误按源逻辑重试。密钥来自环境或 `.env.local` 的
`UNILM_NAV_API_KEY`，也可使用 `DASHSCOPE_API_KEY`。

控制器保留源方法：位置误差小于 5 cm 后锁定到达，停 1 s，再完成最终朝向；
朝向容差 7°、稳定 1 s。前进 0.4 m/s、转向 0.1–0.3 rad/s，转向与前进分段执行，
目标最大距离 3 m，对齐总时限 180 s。使用 FASTLIO 的实际位姿；到达锁定后不重新接近。

不运行原 YOLOE 对齐角色 grounding 或额外 ALIGN VA。只有控制器返回
`reached/aligned` 才签发当前操作的单次启动凭据；失败、取消、过期结果不能启动 VLA。
后续 VLA 的 readiness、模型切换、完成检查、超时继续及 PLANNER 交接沿用当前 agent。

每次推理图像/深度、请求响应、位姿预测和运动日志保存在该实验 session 的
`unilm_nav/` 中，`events.jsonl` 的 `UNILM_COMPLETED/UNILM_FAILED` 关联输出路径和操作身份。
这些系统状态不替代人工物理成功标注。
