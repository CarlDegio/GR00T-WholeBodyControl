# V7 无编号 + 左上角头部相机对照

最终成图：`teaser_v7_head_insets.png`，1920×1080。移除数字编号及其引线，保留V6白色虚线及转向符号；左上角按“Before alignment / After alignment”上下放置原5、6时刻头部相机画面。完整4:3画面等比缩小，不裁切，不覆盖背景中正在踩桶机器人的头部。

## 画面来源与检测

相机原始数据：`outputs/experiments/full_vln_T4_visible_01_0fce0b283641/demo/0fce0b283641-g1/video/head.mjpeg`，按`head.jsonl`中的source_timestamp_ns匹配对应视频时刻，通过offset/length直接读取原始JPEG。记录视频head.mp4也已检查，这两个时刻没有目标/对齐线叠加；叠加状态见head_frame_selection.json。

实际提示词：`green rubbish bin`。目标和yaw-align目标相同，来源为原试验事件e3770bb10aae4198808955fce6bc977b的va_align_grounding结果，证据保存为prompt_log_evidence.json。

在两张相机原图上运行本地`yoloe-26m-seg.pt`文本提示分割，CPU，imgsz=640，conf=0.25，iou=0.5。绿色框/轮廓/轻微掩码来自实际模型输出。6号红线来自项目现有RGB掩码交集+Canny/Hough线段检测，选最长候选；没有深度筛选，不能当作当时在线选线或测量yaw。5按用户允许不展示边缘线。所有新增标注都是离线推理，不是在线记录。

| 时刻 | 对应第三人称视频秒 | 最近头部源帧时间差 | YOLOE置信度 |
|---|---:|---:|---:|
| 5 | 287.566667 | +0.030263s | 0.8240 |
| 6 | 331.820000 | +0.015724s | 0.8783 |

`head_5_raw.jpg`和`head_6_raw.jpg`为原始相机图；`head_5_annotated.png`和`head_6_annotated.png`为离线检测标注图；`head_insets_full_resolution.png`为原分辨率上下对照。检测框、掩码、线端点、候选线及来源均有单独文件留存。既有整秒视频同步映射精度限制仍适用。

验证：小图及阴影区域以外像素与V6无编号版本相同；两幅相机原图均640×480；两次实际YOLOE检出；6实际边缘线检出；最终1920×1080。未修改任何实验事件或结果统计。
