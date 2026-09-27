# 四任务 Demo · V6 替换 T2 并按指定时间裁切

## 当前导出

- `exports/02_alignment_operations_2x2_v6.mp4`：最新对齐＋操作页。
- `exports/00_combined_preview_v6.mp4`：最新两页合并。
- `exports/01_operations_2x2_v2.mp4`：纯操作页，沿用原文件，未改变。

沿用 1920×1080、60 fps、2 倍速、无音轨及既有四宫格：T1 左上、T2 右上、T4 右下，T3 留空。任务播完后显示结束卡。

## 当前裁切

- T1 对齐＋操作：原片末尾累计去掉 2 秒，保留 1873 帧。
- T4 对齐＋操作：原片末尾去掉 2 秒，保留 1544 帧。
- T2 对齐＋操作：使用新素材 `1290e0e88e03cbccca3fde45e30d94df.mp4`，删除原视频 `[0,8)` 秒与 `[36,46)` 秒，仅保留 `[8,36)` 秒和 `[46,结束)`。原片 2701 帧，保留 2161 帧，2 倍速播放约 36.017 秒。两段直接拼接，不叠化或重叠；首帧对应原片 8 秒，切口从原片 36 秒前一帧接原片 46 秒。
- 四条纯操作素材不裁切，原文件复用。

长片全程标注“对齐＋操作”，未根据画面推断系统阶段切换。除用户明确指定时间外不另行删除停顿、开头、中间或结尾。

## 素材库

原视频实际副本位于 `assets/originals/`，来源、当前路径和 SHA-256 记录在 `assets/manifest.json`。最新 T2 为第 8 条，状态 active；第 7 条旧 T2 标记为 superseded，保留历史副本，不参与当前渲染。第 1–6 条沿用之前素材。素材库更新不修改实验结果或原实验任务编号。

## 复现与检查

在项目根目录执行：

```sh
.venv_inference/bin/python demo_video/2026-09-25/edit/render_clips.py
.venv_inference/bin/python demo_video/2026-09-25/edit/render_pages.py
.venv_inference/bin/python demo_video/2026-09-25/edit/verify_v6.py
```

`edit/edit_decisions.json` 是当前有效剪辑规则。`edit/verification.json`、`edit/v6_change_check.json` 核对 T2 帧数、首帧、切口两侧帧、尾帧在最终四宫格中的对应关系，以及 T1/T4 中间视频、纯操作页哈希不变和成片完整解码。旧剪辑规则与清单位于 `versions/`，旧版导出保留版本后缀。
