# 四任务 Demo · V8 T2 两条操作改为 4 倍速

## 当前导出

- `exports/02_alignment_operations_2x2_v6.mp4`：最新对齐＋操作页。
- `exports/00_combined_preview_v8.mp4`：最新两页合并。
- `exports/01_operations_2x2_v8.mp4`：纯操作页，新增 T2 右上角，两条按上传顺序播放，T2 为 4 倍速；T1/T4 仍为 2 倍速。

沿用 1920×1080、60 fps、无音轨及既有四宫格；T2 纯操作为 4 倍速，其余为 2 倍速。布局：T1 左上、T2 右上、T4 右下，T3 留空。任务播完后显示结束卡。

## 当前裁切

- T1 对齐＋操作：原片末尾累计去掉 2 秒，保留 1873 帧。
- T4 对齐＋操作：原片末尾去掉 2 秒，保留 1544 帧。
- T2 对齐＋操作：使用新素材 `1290e0e88e03cbccca3fde45e30d94df.mp4`，删除原视频 `[0,8)` 秒与 `[36,46)` 秒，仅保留 `[8,36)` 秒和 `[46,结束)`。原片 2701 帧，保留 2161 帧，2 倍速播放约 36.017 秒。两段直接拼接，不叠化或重叠；首帧对应原片 8 秒，切口从原片 36 秒前一帧接原片 46 秒。
- T1/T4 四条纯操作素材不裁切，既有拼接视频复用。
- T2 两条新操作素材（第 9、10 条）完整时间范围保留，按上传顺序拼接，4 倍速；无开头、中间或结尾裁切。为保持 60 fps，按 4 倍速正常抽帧，保留原始首尾帧。对应两段约 13.450 秒及 9.217 秒。

长片全程标注“对齐＋操作”，未根据画面推断系统阶段切换。除用户明确指定时间外不另行删除停顿、开头、中间或结尾。

## 素材库

原视频实际副本位于 `assets/originals/`，来源、当前路径和 SHA-256 记录在 `assets/manifest.json`。当前 T2 对齐＋操作为第 8 条，状态 active；第 7 条旧 T2 标记为 superseded，保留历史副本，不参与当前渲染。第 1–6 条沿用之前素材。第 9、10 条为本次两条 T2 纯操作，已保存实际副本及哈希。素材库更新不修改实验结果或原实验任务编号。

## 复现与检查

在项目根目录执行：

```sh
.venv_inference/bin/python demo_video/2026-09-25/edit/render_clips.py
.venv_inference/bin/python demo_video/2026-09-25/edit/render_pages.py
.venv_inference/bin/python demo_video/2026-09-25/edit/verify_v8.py
```

`edit/edit_decisions.json` 是当前有效剪辑规则。`edit/verification.json`、`edit/v8_change_check.json` 核对两条 T2 操作的 4 倍速时长、首尾帧及两条拼接顺序，以及 T1/T4 操作中间视频、对齐＋操作页哈希不变和成片完整解码。旧剪辑规则与清单位于 `versions/`，旧版导出保留版本后缀。
