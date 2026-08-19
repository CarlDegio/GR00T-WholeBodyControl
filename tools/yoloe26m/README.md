# YOLOE-26M 本地评估

这里使用 Ultralytics 官方 yoloe-26m-seg.pt，支持开放词汇文本提示、目标框和实例分割掩码。

本机配置复用了仓库已有的 .venv_inference：

- Python 3.10.20
- PyTorch 2.7.1+cu128
- TorchVision 0.22.1+cu128
- Ultralytics 8.4.117
- RTX 5070 Laptop 的 sm_120 原生 CUDA 支持

下载的模型、MobileCLIP2 文本编码器、样图、Ultralytics 配置和输出都在
tools/yoloe26m/ 下，并已由该目录的 .gitignore 排除。

## 重建或检查安装

~~~bash
cd /home/wang/projects/GR00T-WholeBodyControl
bash tools/yoloe26m/setup.sh
~~~

当前脚本复用已有 CUDA PyTorch，通常不属于十分钟以上的大下载。如果以后改成
全新环境并需要重下数 GB PyTorch/CUDA 包，再这样放进 tmux：

~~~bash
tmux new-session -s yoloe26m_setup
bash tools/yoloe26m/setup.sh 2>&1 | tee tools/yoloe26m/setup.log
~~~

按 Ctrl-b d 离开会话；用 tmux attach -t yoloe26m_setup 返回。

## 文本提示推理

每个新终端先执行：

~~~bash
cd /home/wang/projects/GR00T-WholeBodyControl
source .venv_inference/bin/activate
~~~

图片：

~~~bash
python tools/yoloe26m/infer.py path/to/image.jpg \
  --classes person cup bottle "cardboard box" \
  --conf 0.25 \
  --name image_test
~~~

视频：

~~~bash
python tools/yoloe26m/infer.py path/to/video.mp4 \
  --classes person chair table cup \
  --name video_test
~~~

0 号摄像头，按 Ctrl-C 停止：

~~~bash
python tools/yoloe26m/infer.py 0 \
  --classes person chair cup \
  --show \
  --name camera_test
~~~

提示词建议使用简短的英文类别名。标注后的图片或视频保存在
tools/yoloe26m/outputs/。模型输出同时包含检测框和像素级实例掩码。

### 用当前采集数据测试桌子检测

以下命令会自动选择 `outputs/base_pose_adjustment/` 下最新一组
`review_samples/raw` 实机 RGB 帧，分别用多个文本 prompt 检测桌子：

~~~bash
source .venv_inference/bin/activate
python tools/yoloe26m/test_text_prompt_table.py
~~~

也可以指定数据目录、提示词、阈值和采样数：

~~~bash
python tools/yoloe26m/test_text_prompt_table.py \
  --source /path/to/rgb_frames \
  --prompts table desk "wooden table" \
  --conf 0.10 \
  --max-images 30
~~~

结果保存在 `tools/yoloe26m/outputs/text_prompt_table_<时间>/`：`report.md` 和
`summary.json` 汇总各 prompt 的逐帧检出率、置信度与耗时；每个 prompt 子目录还包含
逐帧结构化检测结果、标注图及 `contact_sheet.jpg`。当前采集数据没有人工桌子标注，
因此检出率只能用于 prompt 筛选，不能当作 precision、recall 或 mAP。

## 性能测试

~~~bash
python tools/yoloe26m/benchmark.py --warmup 10 --runs 100
~~~

基准不包含模型加载和一次性的文本提示编码，报告热态端到端延迟、模型推理延迟、
吞吐量和峰值分配显存。

## 本机已测结果

2026-08-10，在 RTX 5070 Laptop、640px、FP16、batch=1、person/bus 文本提示下，
预热 10 次后正式运行 50 次：端到端平均 9.65 ms（103.7 FPS），中位数 9.11 ms，
p95 11.95 ms；模型推理平均 8.03 ms；峰值分配显存约 154 MiB。官方 bus.jpg
样图得到 5 个 person、1 个 bus，以及对应的 6 个实例掩码。这个数字不包含模型加载和
一次性的文本提示编码，视频解码、保存和显示会降低实际整条流水线帧率。

## 常用参数

~~~text
--device 0          使用 0 号 CUDA GPU，检测到 CUDA 时为默认值
--device cpu        强制 CPU 推理
--imgsz 640         输入分辨率；提高它可能改善小目标召回，但速度和显存占用会增加
--conf 0.25         置信度阈值；漏检多时可降到 0.15，误检多时可升到 0.35
--no-half           禁用 GPU FP16，用于排障
--max-frames 300    摄像头或流处理 300 帧后停止；0 表示不限制
--exist-ok          复用指定输出目录
~~~

导出 ONNX 或 TensorRT 后，文本类别会被固化在模型中；要修改类别，需要从原始
PyTorch 权重重新设置提示并再次导出。

## 参考图片 + 示例框

这套接口只处理两张静态图片：`--refer-image` 是第一张参考图，位置参数是随后要
识别的第二张图。手工模式的 `--bbox` 使用参考图原始像素坐标
`X1 Y1 X2 Y2`，可以重复多次；同一目标的所有示例框都会作为 YOLOE 视觉类别 0，
不会只保留第一个。`--target-name` 负责明确类别含义和输出标签。

完整命令 1：手工给第一张图的示例框。

~~~bash
cd /home/wang/projects/GR00T-WholeBodyControl
source .venv_inference/bin/activate
python tools/yoloe26m/visual_prompt.py /absolute/path/to/second.jpg \
  --refer-image /absolute/path/to/first.jpg \
  --target-name "blue basket" \
  --bbox 120 80 360 300 \
  --bbox 500 100 720 330
~~~

## 首层模型自动生成参考框

自动模式要求显式传入 `--target`，所以你可以直接设定首层模型寻找什么。首层模型
只看第一张参考图，并返回该目标的所有可见实例框；脚本验证其 `[0,1000]` 归一化
坐标、换算为原图像素后，不暂停地让 YOLOE 识别第二张图。首层返回
`NOT_FOUND`、`UNSURE`、空框、越界框或错误目标时，脚本会在加载 YOLOE 前报错退出。

完整命令 2：Qwen-VL Plus。它复用 BasePose 的 DashScope OpenAI-compatible API、
模型和 key 查找逻辑：优先读取 `DASHSCOPE_API_KEY`，否则读取
`.venv_inference/.env`。

~~~bash
cd /home/wang/projects/GR00T-WholeBodyControl
source .venv_inference/bin/activate
python tools/yoloe26m/auto_refer_detect.py /absolute/path/to/second.jpg \
  --refer-image /absolute/path/to/first.jpg \
  --target "blue basket"
~~~

每次运行自动写入 `tools/yoloe26m/outputs/<target-slug><number>/`。手工和 Qwen
共用同一个编号序列；例如已有 `table1` 时下一次 table 任务创建
`table2`，已有 `table1` 和 `table3` 时创建 `table4`。`blue basket` 会转换成
`blue_basket1`。已有目录永远不会被覆盖或复用。

每个编号目录包含：

- `reference_annotated.jpg`：第一张图及所有示例框；
- `target_annotated.jpg`：第二张图的 YOLOE bbox、类别、置信度和实例 mask；
- `reference_prompt.json`：最终送给 YOLOE 的原图像素框；
- `detections.json`：第二张图的 bbox、置信度与 mask 数量；
- 自动模式还保存 `grounding_prompt.txt`、`grounding_schema.json` 和
  `grounding_response.json`，但不会保存 key 或模型的隐藏推理内容。

如果只找到一个目标，自动首层只产生一个示例框；如果找到多个，则全部传给 YOLOE。
