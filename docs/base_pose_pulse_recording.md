# BasePose vy 脉冲记录

自己的 YOLOE BasePose（独立 B 启动、Full ObjectNav/VLN 的 ALIGN）默认记录。
实验文件为该次运行目录下的 `vy_pulses.jsonl`；独立运行文件为
`outputs/base_pose_adjustment/vy_pulses_<启动时间戳>.jsonl`。
READY 事件显示绝对路径。精简日志模式也保存，每个实际发出的脉冲一行。

- `vy_m_s`：发送的 vy，保留正负号。
- `pulse_duration_s`：从首次非零脉冲指令发出，到首条停止/替代指令发出的实际间隔。
- `wait_duration_s`：随后零速等待的观测窗口，正常为 0.5 秒；提前运动或取消会截短并记录原因。
- `pulse_displacement`、`wait_displacement`、`total_displacement`：脉冲、等待、合计的视觉净位移，单位米。
  `planar_m` 为平面位移大小，`left_m` 为有符号横移（左正右负），`forward_m` 为前向位移。
- `visual_samples`：三个边界使用的目标反投影坐标和图像时间戳，保留插值跨度以便判断时间分辨率。

测量复用当前视觉控制所用的未滤波 `TargetGeometry.forward_m/right_m`：
分割 mask 深度和 bbox 横向中心经过相机内参反投影、相机到身体坐标变换。
在脉冲开始、结束、等待结束的时刻，用前后图像坐标线性插值。
机器人前向位移估计为 `forward_start - forward_end`，左向位移为 `right_end - right_start`。
不读取里程计位置，也不以速度乘时间代替测量。

这是相对静止目标的视觉位移估计，假设窗口内相机朝向变化很小；深度、检测框和机身晃动会影响数值。
切相机、换目标、跟踪中断、缺少边界图像或帧间隔过大时写 `missing_reason`，不填零。
`vy_pulses.jsonl` 只保存脉冲摘要及三个边界样本，文件写入在独立线程执行。

自己的 BasePose 服务同时恢复原有非图片日志，即使 ObjectNav 实验设置了
`SONIC_EXPERIMENT_MINIMAL_LOGGING=1` 也会保存：

- `outputs/logs/inference/base_pose.log`：原有运行状态、控制阶段、告警和错误文字日志（沿用原有轮转）。
- `outputs/base_pose_adjustment/dual_raw_yoloe_<时间>_g<generation>/raw_servo_frames.jsonl`：
  原有逐帧检测框、跟踪 ID、反投影几何、候选边线、控制器状态、滤波误差、脉冲状态、速度命令和姿态信息。
  根目录可通过 `output_root` 修改，READY 事件包含 `frame_log_root`。
- 实验目录中的 `events.jsonl` 和 `vy_pulses.jsonl` 继续保存。

BasePose 服务将诊断图片采样间隔设为 0，既不保存初始诊断图，也不保存逐帧诊断图；
上述逐帧 JSONL 仍正常写入。ObjectNav 自身的 grounding/postcheck 证据图不属于此开关。
记录不调整任何运动参数，下次启动自己的 BasePose 时生效；过去未保存的详细日志无法补回。
