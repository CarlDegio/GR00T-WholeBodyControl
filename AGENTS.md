# 实验数据归档约定

用户已指定助手负责本项目的实验数据标注、归档及成功率汇报。

- 完成每次用户要求的标注后，将标注追加到原始 `outputs/experiments/<session>/events.jsonl`，再运行 `.venv_inference/bin/python tools/archive_experiment_data.py`，同步 `experiment data/` 中对应 T1–T4 任务的数据、索引与统计。该流程已获用户授权，无需重复询问。
- 正式范围沿用 `outputs/experiments/formal_results/selection_policy.json`：北京时间 2026-09-08 18:00 起，按 trial_start 筛选。最新 smoke / exclude_reason 标注优先；有效数据包括正式成功、失败与待标注。
- 使用最新独立成功/失败标注；系统 reached / manipulation_completed 不等同于人工成功。不得补造物理完成时间。保留用户的条件编号修正。
- 按 trial_id 唯一归档，复制真实文件并保留原始数据。不要同时汇总原始日志和归档副本，不要把 withdrawn 历史数据重复计入。
- 每次汇报本批与累计的成功数、已标注分母、成功率、smoke 和待标注数量；按任务、方法分别列出。用户于 2026-09-12 要求将 Full ObjectNav 从统计和表中移除，遵循 selection_policy.json 的 excluded_methods，后续汇总不再列出或计入 full_objectnav。原始数据保留，已归档副本移入 withdrawn。待标注不视为失败。
- Progress 必须保留用户提供的分母。`result --progress N --progress-denominator D` 可记录显式分数；汇总导出原始分子、分母及比例。2026-09-12 的 F1–F6 已按 0/3、0/3、1/3、2/3、0/3、0/3 记录，三阶段名称及是否统一全部 T1 尚待用户回答；不要擅自将旧冻结配置改成三阶段，也不要把 F4 的 2/3 当成 2/2。
- 几何对齐 T1 / visible_01 按用户选择，仅保留 2026-09-09 的成功试验 `3b033d7d3109-g1`；2026-09-08 的失败试验 `b3df1b125f7e-g1`（F1）已标为 smoke test，其历史 Progress 0/3 不再进入正式统计。
- 用户于 2026-09-13 确认，后续统计和表格移除 NaVILA（`navila_basepose_vla`），改用 `baselinelavira`；遵循 selection_policy.json 的 excluded_methods、method_replacements 和 report_methods。此次用户明确要求“暂不开始统计，只确认”，因此只记录口径并核对日志，没有刷新数值表或热力图；下一次用户要求统计或标注时应用新口径并正常同步归档。BaselineLAVIRA 当时有两条原始试验记录，其中只有 `2a029a0f3b0f-g1` 进入 VLA；`a54cd75a19a5-g1` 在 VA 校验时终止。未替用户新增成功、失败或 smoke 标注，不得仅因未进入 VLA 自动排除。
