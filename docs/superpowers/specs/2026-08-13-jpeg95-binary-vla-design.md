# JPEG 95、二进制相机传输与 VLA 单次解码设计

日期：2026-08-13

分支：`experiment/jpeg-quality-95`

## 1. 背景

当前相机链路把软件编码的 JPEG/PNG 转成 Base64 字符串后放入 msgpack。VLA 链路又把 Gateway 解码得到的 RGB 图像重新编码为 JPEG，再由 OpenPI 解码：

```text
Camera JPEG -> Base64/msgpack -> Gateway JPEG decode -> RGB shared memory
             -> VLA JPEG encode -> msgpack -> OpenPI JPEG decode
```

这带来两类额外开销：

- Base64 使相机图像字段膨胀约 33%；当前真实 JPEG 95 数据包从约 618,635 字节增加到 824,674 字节。
- VLA 在 Gateway 已解码后再次编码。真实四相机帧基准中，二次 JPEG 编码和请求打包平均耗时约 3.41 ms。

本设计把所有生产 JPEG/MJPEG 质量统一为 95，并让 VLA 通过 Gateway 获取相机原始 JPEG 字节，由 OpenPI 只解码一次。

## 2. 目标

1. 所有生产链路的 JPEG/MJPEG 编码质量统一为 95。
2. 相机 ZMQ 包内的 JPEG 和 PNG 使用 msgpack binary，不再由生产端 Base64 编码。
3. Gateway 同时提供：
   - `camera_encoded/*`：原始 JPEG 字节，供 VLA 使用；
   - `camera/*`：解码 RGB/深度共享内存，供其他消费者使用。
4. VLA 形式上仍只访问 Gateway，不直接订阅相机 ZMQ。
5. 保持当前 `VlaSensorGatewayIngress` 单后台循环及顺序轮询方式，不新增独立线程、相机/状态配对器或调度策略。
6. OpenPI 对 VLA 图像只做一次 JPEG 解码。
7. 量化计算延迟、帧率、吞吐和丢帧/陈旧帧情况。

## 3. 非目标

- 不改变其他消费者读取 `camera/*` RGB 共享内存的方式。
- 不让 VLA 直接连接相机 PUB 端口。
- 不增加 VLA 相机线程、状态线程或新的跨流时间配对逻辑。
- 不改为 ZMQ multipart，不新增端口。
- 不在本轮并行解码 OpenPI 的四幅 JPEG。
- 不改变模型预处理、推理频率或动作控制逻辑。
- 不启动完整模型做协议验证；使用真实相机包完成端到端协议测试。
- PNG 深度仍为无损编码，不引入“质量 95”概念。
- 不修改纯测试/基准工具中为了覆盖其他质量而保留的参数值。

## 4. 最终数据流

```text
Sonic Camera Server
  JPEG/PNG raw bytes in one msgpack message
              |
              v
SensorGateway CameraZmqIngress
  1. unpack message
  2. publish RGB JPEG bytes -> camera_encoded/*
  3. synchronously decode the same payload
  4. publish RGB/depth arrays -> camera/*
              |
              +--------------------------+
              |                          |
              v                          v
VlaSensorGatewayIngress             Existing consumers
  sequential camera/state polls       camera/* RGB/depth
  camera_encoded/*
              |
              v
run_vla_inference
  wrap existing JPEG bytes; no cv2.imencode
              |
              v
OpenPI policy server
  decode each JPEG once -> RGB -> existing preprocessing/model path
```

Gateway RPC 已运行在其现有独立服务线程中。因此主 Gateway 线程同步解码 `camera/*` 时，VLA RPC 可以并发读取已经优先发布的 `camera_encoded/*`；VLA 自身不需要增加线程。

## 5. 协议与兼容性

### 5.1 相机 ZMQ 消息

- 保持单个 msgpack 消息和现有字段结构。
- 新生产端把 RGB JPEG 和深度 PNG 写为 msgpack binary (`bytes`)。
- 消息同时携带每路原始图像 shape；这是编码前已有的数据描述，不做 JPEG
  解析、解码、时序判断或额外等待。
- `ImageMessageSchema` 已允许 binary 图像字段，因此不强制提升 schema version。
- 普通 `ImageMessageSchema.deserialize` 继续接受历史 Base64 字符串，仅用于非
  VLA 消费者读取旧数据；这不是 VLA 滚动升级或回退保证。
- 新软件 JPEG 编码前显式执行 RGB 到 BGR 转换，再调用 OpenCV 编码；binary 解码后显式转换回 RGB，保证颜色语义一致。
- OAK 设备产生的 MJPEG 字节直接透传。

### 5.2 Gateway encoded stream

每个 `camera_encoded/*` 样本包含一维 `uint8` JPEG 数据、原始图像 shape 及现有时间戳/序号元数据。VLA 编码侧只接受：

- `jpeg_bytes`：新 binary 生产路径；

Gateway 可为普通历史解码消费者标记 `base64_jpeg`，但 VLA ingress 必须拒绝
该编码，不做 Base64 decode、shape inference、fallback decode/re-encode 或颜色修正。

如果 encoded 流缺失、类型错误、超时或四相机时间偏差超过现有阈值，VLA 沿用当前“快照不可用则跳过本轮推理”的行为，不静默退回 RGB 重编码。

### 5.3 VLA 请求

- 继续使用 OpenPI 已支持的 `__opencv_jpeg_rgb__` 标记，不修改 OpenPI 代码。
- VLA 把相机原始 JPEG bytes 与 Gateway `camera_info` 中的 width/height 封装为
  现有 shape/dtype/data 格式；OpenPI 的现有解码器仍只解码一次。
- `run_vla_inference` 删除实时链路中的 `cv2.imencode`，仅封装 bytes 并执行 msgpack 打包。

## 6. 调度和时序

本轮严格保持原有 VLA 调度：

1. `VlaSensorGatewayIngress` 的单个后台 worker 先轮询四路相机快照，再轮询机器人状态。
2. 四路相机仍使用现有序号、时间戳和最大 5 ms 偏差检查。
3. 相机和机器人状态不新增严格配对、等待窗口或重采样。
4. `run_vla_inference` 继续读取该缓存并按现有频率发起推理。

因此，本次优化只删除数据转换和排队开销，不改变已运行系统的时序语义。

实施硬约束：`VlaSensorGatewayIngress._run`、`_request`、`_poll_state`、现有
age/skew/sequence 检查及轮询顺序保持不变。VLA 热路径不新增时间戳比较、等待
窗口、重试、sleep、数据配对或调度分支；encoded adapter 仅验证
`encoding == "jpeg_bytes"` 并透传 JPEG bytes。

## 7. 降低相机传输延迟

- 相机 PUB socket 的发送高水位调整为 latest-first 所需的小队列（目标 `SNDHWM=1`），保留非阻塞发送；消费者落后时优先丢弃旧帧。
- Gateway SUB 继续使用 `CONFLATE=1` 和 `LINGER=0`。
- Gateway 在做任何 JPEG 解码前发布 `camera_encoded/*`。
- 不使用 multipart、额外复制队列或新的接收线程。

该策略优化的是“最新帧延迟”，而非保证每帧必达；这与实时 VLA 和当前 CONFLATE 语义一致。

## 8. JPEG 95 生产范围

以下生产默认值或显式启动参数改为 95：

- 物理相机/组合相机的软件 JPEG；
- OAK MJPEG；
- Gemini 启动链路；
- MuJoCo 仿真相机；
- NavDP JPEG；
- 可视化 JPEG；
- 其他调用生产 schema 默认质量的相机服务。

`start_camera_server.zsh` 已显式使用 95，继续保留。VLA 不再拥有二次编码质量参数，因为不再进行二次编码。

## 9. 测量方法与指标

### 9.1 采集周期

在 Sonic 相机和 PC Gateway 稳定后连续采集 60 秒，分别记录：

- Camera Server 发送消息 FPS；
- Gateway 接收消息 FPS；
- 四路 `camera_encoded/*` 的 Gateway publication FPS 与 producer source timestamp
  物理唯一帧 FPS；
- 六路 `camera/*` 的 Gateway publication FPS 与物理唯一图像 FPS；
- 六路物理唯一图像合计 images/s；
- 相机 ZMQ payload 的 MiB/s 和 Mbit/s；
- VLA 请求 FPS、bytes/request 和 Mbit/s；
- `zmq.Again` 发送 API 失败、Gateway 序号跳变、source timestamp 重用/回退/缺失、
  陈旧帧和 Gateway 快照拒绝次数。没有独立 producer frame ID 时，PUB/HWM
  transport drop 不可观测，不以 send API 或 Gateway sequence 推断为零。

计算方式：

```text
physical FPS = unique positive producer source timestamp count / elapsed seconds
Gateway publication FPS = unique Gateway ring sequence count / elapsed seconds
MiB/s = total payload bytes / elapsed seconds / 2^20
Mbit/s = total payload bytes * 8 / elapsed seconds / 10^6
images/s = sum(unique frames for all image streams) / elapsed seconds
```

### 9.2 计算延迟

使用 `perf_counter_ns` 记录并输出 mean、P50 和 P95：

- Camera Server：JPEG/PNG 编码、msgpack serialize、ZMQ send；
- Gateway：recv/unpack、encoded publish、JPEG/PNG decode、decoded publish；
- VLA：Gateway camera RPC、JPEG prepare/wrap、msgpack pack、request round trip；
- OpenPI：四幅 JPEG 顺序解码，以及不含模型推理的请求预处理总时间。

VLA telemetry 中原 `jpeg_encode` 分段更名为 `jpeg_prepare`，避免把字节封装误报为编码耗时。

### 9.3 已有基线

同一设备上的 JPEG 95 基线：

- 相机帧率：30.004 FPS；
- Base64 相机吞吐：192.613 Mbit/s；
- 相机到 PC 延迟均值：45.517 ms；
- PC 端六图解码均值：10.483 ms。

真实四 RGB 帧微基准：

- 旧 VLA 二次编码和请求打包：mean 3.407 ms，P95 3.773 ms；
- 新直接 JPEG 封装和请求打包：mean 0.011 ms，P95 0.011 ms；
- OpenPI 四 JPEG 顺序解码：mean 4.291 ms，P95 4.564 ms；
- 旧 codec 计算路径约 7.70 ms，新路径约 4.29 ms，预计减少约 44%。

当前 2 Hz VLA 请求的图像吞吐约从 5.60 Mbit/s 降至 5.55 Mbit/s；主要收益是计算延迟，而不是 PC 到 OpenPI 的包大小。

## 10. 验收标准

### 10.1 功能

- 所有列出的生产 JPEG/MJPEG 编码质量为 95。
- 新相机生产包中的 RGB JPEG 和深度 PNG 都是 msgpack binary。
- `camera_encoded/*` 在对应 `camera/*` 解码发布之前可用。
- VLA 只从 Gateway 获取四路 JPEG，不直接订阅相机。
- VLA 实时路径不调用 `cv2.imencode`。
- OpenPI 现有协议每幅图只调用一次 JPEG decode，OpenPI 仓库无改动。
- 普通 schema decoder 仍可读取历史 Base64 相机包；VLA encoded ingress 只接受
  `jpeg_bytes`，并继续输出既有 VLA JPEG marker。
- RGB 色彩测试能识别红/蓝通道，不发生静默 BGR/RGB 互换。

### 10.2 性能

- 60 秒 Camera Server/Gateway 消息 FPS 不低于 29。
- 每路 `camera_encoded/*` producer source timestamp 物理唯一帧 FPS 不低于 29。
- 每路正常工作的 `camera/*` 物理唯一帧 FPS 不低于 29，六路物理唯一图像
  合计不低于 174 images/s。Gateway publication FPS 单独报告，不代替物理阈值。
- 对同一批消息，以实际 binary 大小和“若使用 Base64”的反事实大小比较，相机 ZMQ payload 至少减少 20%；预期约 25%。
- `jpeg_prepare` P95 小于 1 ms。
- 在同一 PC 上 OpenPI 四幅顺序 JPEG decode P95 不高于 6 ms。
- 60 秒内没有因 socket 队列累积产生持续增长的帧龄；报告 send API failures、
  source timestamp reuse/discontinuity、observed Gateway sequence gaps 和 snapshot
  rejections。无 producer ID 时不宣称 PUB/HWM transport drop 为零。

性能阈值用于检测回归。相机到 PC 的绝对延迟受时钟同步和现场网络影响，同时报告原始结果，不以单次均值作为唯一通过条件。

## 11. 测试策略

### 11.1 单元测试

- schema 对 JPEG/PNG binary 的生产序列化及普通 decoder 的历史 Base64 解码；
- VLA encoded ingress 对 `base64_jpeg` 明确报 unsupported encoding；
- RGB/BGR 色彩语义；
- Gateway encoded-first 发布顺序；
- VLA encoded stream 校验、四相机偏差和缺帧行为；
- `run_vla_inference` 直接封装 JPEG，使用 monkeypatch 证明未调用 `cv2.imencode`；
- VLA 用相机原始 JPEG 构造 OpenPI 现有 marker 的 shape/dtype/data 字段。

### 11.2 协议集成测试

保存一组 Sonic JPEG 95 真实消息，执行：

```text
camera msgpack
  -> CameraZmqIngress
  -> camera_encoded/* and camera/*
  -> VlaSensorGatewayIngress
  -> VLA request msgpack
  -> OpenPI request decoder
  -> four RGB arrays
```

测试比较最终 RGB 的形状、通道、帧标识和时间戳，并统计各阶段耗时/字节量。此测试不加载完整 VLA 模型。

### 11.3 现场测试

- 启动 Sonic 相机 `zsh` 脚本并确认显式 JPEG 95；
- 启动 PC Gateway 和协议测试客户端；
- 运行 60 秒 FPS/吞吐/延迟采集；
- 保存机器可读 JSON 和简短 Markdown 报告；
- 与已提交 JPEG 80/95 基线并列比较。

## 12. 部署和回退边界

VLA 不提供 Base64/binary 混合版本兼容窗口。启动 VLA 前必须同时确认 PC
Gateway 输出四路 `jpeg_bytes` 且 Sonic producer 使用 binary msgpack/JPEG 95；
现有 OpenPI marker 保持不变。普通 schema decoder 的历史 Base64 能力只服务非
VLA 旧数据读取，不能作为 VLA 回退路径。若 producer 回退到 Base64，VLA 必须
停止并报告 unsupported encoding，而不是静默 decode/re-encode。

## 13. 代码库与分支边界

- `GR00T-WholeBodyControl` 的改动继续位于独立分支 `experiment/jpeg-quality-95`。
- `openpi_sonic` 保持 `origin/main` 内容不变，不创建功能提交，不触碰主工作区中现有未跟踪的 `replay_data/`。
- 现场启动/停止操作单独记录。
