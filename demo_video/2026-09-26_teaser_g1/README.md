# G1 teaser 合成预览

成图：`teaser_v2_preview.png`。使用内置 imagegen 生成式合成，不是逐像素忠实的传统抠图；机器人局部外观、姿态与投影位置仍有生成偏差，仅供版式效果预览，不作为测量图。虚线为关键帧连接示意，不是测量轨迹。

原始六帧：`frames_synced/01.png`–`06.png`；对照图：`six_frames_synced.png`。

同步来源：`outputs/demo_edits/T4_20260925_v1/README.md`，视频零点为 2026-09-25 21:20:19 CST；该映射只有整秒精度，不保证跨源逐帧同步。

日志：`outputs/experiments/full_vln_T4_visible_01_0fce0b283641/events.jsonl`，trial `0fce0b283641-g1`。阶段采用 SKILL_STARTED / SKILL_COMPLETED。操作1完成事件是 timeout_continued，不等同独立物理成功标注。未修改实验记录或统计。

| 编号 | 时刻 | 视频 PTS（秒） | 0起始帧号 |
|---|---|---:|---:|
| 1 | 起始 | 0.000000 | 0 |
| 2 | 对齐1开始前一帧 | 93.200000 | 2796 |
| 3 | 对齐1结束后一帧 | 129.166667 | 3875 |
| 4 | 操作1结束后一帧 | 180.800000 | 5424 |
| 5 | 对齐2开始前一帧 | 287.566667 | 8627 |
| 6 | 对齐2结束后一帧 | 331.833333 | 9955 |

取帧规则：开始前选择 PTS 严格小于事件时间的最后一帧；结束后选择严格大于事件时间的第一帧。完整事件及纳秒时间见 `frame_manifest.json`。

实际提示词：`prompt_final.txt`；第二轮修正：`prompt_correction.txt`。`frames/` 和其余 review 文件仅为早期目测探索，不是最终选帧。
