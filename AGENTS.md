# 实验数据归档约定

用户于 2026-09-12 指定本项目助手同时负责实验数据标注、归档和成功率汇报。

- 归档根目录为项目根目录下的 `experiment data/`，按 T1–T4 分别保存。任务名称与操作方式见该目录的 `README.md`。
- 每次完成用户要求的标注后，追加标注到原始 `outputs/experiments/<session>/events.jsonl`，随后运行 `.venv_inference/bin/python tools/archive_experiment_data.py`，更新归档、逐试验索引和累计统计。这是用户已授权的后续流程，无需重复确认归档。
- 正式时间窗口遵循 `outputs/experiments/formal_results/selection_policy.json`：北京时间 2026-09-08 18:00 起，按 trial_start 时间筛选，尊重最新 smoke / exclude_reason 标注。正式样本包含成功、失败以及待标注试验；smoke 不进入统计。
- 成功率使用最新独立标注。不得把系统 reached / manipulation_completed 自动改成人工成功，不得用系统完成时间补造物理完成时间。
- 保留用户的条件编号修正，按 trial_id 唯一归档。原始日志及图像保留，归档使用实际副本；不要同时汇总原始数据与归档副本，也不要用递归扫描将已撤出归档的历史记录重复计入。
- 每次汇报本批及累计的成功数、已标注分母、成功率和待标注数，按任务和方法分别列出；ObjectNav 独立成行。待标注不当作失败，smoke 不进分母。必要时附归档索引链接。

# ICLR 场景图修改与上传

- 用户于 2026-09-26 要求：修改 teaser 场景图后，需要像此前一样上传到对应的 Overleaf 项目；不能只保存本地预览。此后续上传已获授权，无需每次重复确认。
- 保留此前的擦除和其他编辑，只修改用户指定的元素；独立场景图与论文实际引用的 teaser 成图应同步更新。当前项目为 `/home/user/Project/Overleaf/6a925da904325a8d60afac0b`。

# ICLR 术语约定

- 用户于 2026-09-26 明确纠正：论文统一使用带连字符的 `loco-manipulation`，句首和标题使用 `Loco-manipulation`；不使用 `locomanipulation` 连写形式，`Mobile manipulation`、`mobile-manipulation` 等形式也按此统一。参考文献的正式标题保留原名。
