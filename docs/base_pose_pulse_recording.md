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

自己的 BasePose 服务同时保存运行日志和采样图片，即使实验设置了
`SONIC_EXPERIMENT_MINIMAL_LOGGING=1` 也会保存：

- `outputs/logs/inference/base_pose.log`：原有运行状态、控制阶段、告警和错误文字日志（沿用原有轮转）。
- `outputs/base_pose_adjustment/dual_raw_yoloe_<时间>_g<generation>/raw_servo_frames.jsonl`：
  原有逐帧检测框、跟踪 ID、反投影几何、候选边线、控制器状态、滤波误差、脉冲状态、速度命令和姿态信息。
  根目录可通过 `output_root` 修改，READY 事件包含 `frame_log_root`。
- 实验目录中的 `events.jsonl` 和 `vy_pulses.jsonl` 继续保存。

所有实验的 BasePose 诊断图片采样间隔统一为 5，重启旧的冻结实验配置时也会启用。
双相机 BasePose 恢复初始诊断图，并在第 0、5、10… 个感知帧分别保存可用的头、胸相机图片：
`outputs/base_pose_adjustment/dual_raw_yoloe_<时间>_g<generation>/diagnostic_images/<相机>/frame_<帧号>.jpg`。
图片叠加目标 mask（紫色）、头部候选边线（黄色）和实际选中线（绿色）；逐帧 JSONL 的
`image_artifacts` 保存对应图片路径及各相机时间戳。单头和几何基线也按每 5 个新感知帧
保存当前使用相机的图片，分别位于同一根目录的 `head_raw_yoloe_*` 和 `geometric_raw_yoloe_*`
目录；几何基线仅标注目标 mask。编码和写盘沿用异步诊断线程。

独立 BasePose 默认同样每 5 帧采样，可通过 `raw_diagnostic_image_interval_frames` 修改，
设为 0 可关闭独立运行的图片采样。READY 事件显示实际采样间隔。grounding/postcheck 证据图继续保存。
图片记录配置在下次启动 BasePose 服务时生效；过去未保存的过程图片无法补回。
