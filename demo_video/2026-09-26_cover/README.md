# Demo 封面 · 2026-09-26

两版均为 1920×1080、60 fps、5.000 秒、H.264，无音频。

- `exports/cover_25_5s.mp4`：3×3 保持 1 秒 → 1 秒连续缩放为 5×5 → 播放 3 秒。
- `exports/cover_16_5s.mp4`：2×2 保持 1 秒 → 1 秒连续缩放为 4×4 → 播放 3 秒。
- 第 2–3 秒标题由模糊渐显，中心两行；第 3–5 秒清晰保持。
- `clips/` 为全部 25 条实际入选片段，各 5 秒，与封面使用的剪辑和倍速一致。
- `gallery.html` 可逐条播放，★ 表示同时入选 16 条版。`selection.csv` / `edit/selection.json` 记录原片起止、倍速、来源哈希及宫格位置。

标题采用当前论文主稿 `/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex`：
Beyond Arrival: Geometric Alignment for Agentic Humanoid Loco-manipulation

25 条构成：指定移动 4 条、T4 环视 1 条；操作 10 条、对齐 10 条。
T1/T3 各操作 3 + 对齐 2；T2/T4 各操作 2 + 对齐 3。
16 条版是其中的子集，保留指定移动与环视 5 条，另含操作 7 条、对齐 4 条。
同一原片可使用不同、不重叠的动作阶段；并非 25 个独立试验。所有片段连续截取，无动作循环或合成补帧。

21 个本次附件：新增 8 条，13 条哈希与库内内容一致；其中两条仅文件名不同。
同时把此前 T4 长程 demo 原片补存入同一素材库，用于指定导航与环视。
素材库：`../2026-09-25/assets/originals/`，索引为 `../2026-09-25/assets/manifest.json`。
库内现有 22 个唯一原始文件（含先前标记 superseded 的历史文件）。本封面不修改实验标注或成功率。

复现：`.venv_inference/bin/python demo_video/2026-09-26_cover/edit/select_and_render.py`，随后运行 `edit/compose.py`、`edit/package.py`、`edit/verify.py`。
