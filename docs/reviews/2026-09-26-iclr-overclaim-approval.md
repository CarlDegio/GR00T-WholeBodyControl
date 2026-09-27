# ICLR 系统泛化性措辞审核结果

用户逐项审批后要求修改。仅执行批准项，未批准的候选不得自动应用。

- G1a：批准，但保留 bridge，弱化 universal。已改为使用 object-relative geometry 作为 navigation 与 manipulation 间的 bridge，并将 resolve 改为 reduce。
- G1b：不修改。
- G2：批准，但保留 seamlessly bridging，使用 initial observation distribution。已修改。
- G3a：批准。首次执行时原句已被并行更新的双参考对象 insight 段落替换，曾暂时保留；用户随后明确拒绝这段额外重写。现已恢复原动机段，仅将第一句改为获批的 `We use object-relative geometry to represent spatial conditions relevant to the manipulation handoff, including distance, lateral offset, and orientation.`，后续原句保持不变。
- G3b：不修改。
- G4：批准，但不要强调 task-specific policies；强调基于 GATE 构建 practical agentic system。已修改。
- G5、G6、G7：不修改。
- G8：批准，已修改实验导语，明确三项实验问题。
- G9a、G9b、G10：不修改。
- G11a：批准，已修改。
- G11b：批准，但不要强调 task-specific manipulation，不写 while retaining，不反复强调 Stretch。已按最新结论段修改；未恢复其他并行编辑已删除的 OVMM 数值句。
- G12a、G12b 及 G13–G16：不修改。
- 此前 task-irrelevant 一句的修正继续保留；R15、R17 的历史保留要求不受本轮影响。

实际六处替换的完整原文及修改后文本见 `2026-09-26-iclr-approved-overclaim-changes.json`。本轮只修改 root.tex；附录、图表、实验数值和技术实现未改动。

同步记录：用户随后要求同步到 Overleaf。独立副本编译通过；同步期间并行提交已包含全部六处修改，因此未重复提交。重新读取远端 `efe86e8` 的 `root.tex`，逐项确认六处批准文本全部存在。

后续纠正：用户要求“这些额外修改不要”。以 `be21746` 为审批前基准，恢复 Introduction 的动机段、第一项贡献及第三项贡献；保留六处已批准替换并应用 G3a 单句修正。不得恢复双参考对象整段 insight 重写、改写后的第一项贡献，或第三项贡献中额外加入的 35.0%→82.5% 结果。恢复后 Introduction 已与“基准 + 批准修改”逐字核对一致。

恢复已上传：远端提交 `b2785b7`。重新读取远端正文，确认 Introduction 与批准版逐字一致，且六处已批准替换均存在。恢复版编译通过。同步时保留了远端新增提交 `c74d40d`。
