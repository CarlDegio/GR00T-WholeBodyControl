# ICLR 论文语言修改审核记录

- 目标：逐批修正语病、语法和相关格式问题；每批由用户审核后再应用。
- 论文目录：`/home/user/Project/Overleaf/6a925da904325a8d60afac0b/`。
- 本次核对的 Git HEAD：`5055831eafb75b67eb646dfb6e6118df3b1c7822`。
- 本文是建议稿，不代表论文已经修改。应用前须重新核对当前源文，保留其他并行修改。

## 第 1 批：摘要（用户全部通过，已应用）

位置：[root.tex:87](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:87)。

| 编号 | 原文 | 建议 | 性质 |
|---|---|---|---|
| A1 | `with limited cross task / scene generalization` | `with limited generalization across tasks and scenes` | 表达修正 |
| A2 | `we develop a agentic system` | `we develop an agentic system` | 冠词修正 |
| A3 | `with substantially higher success rate than baselines` | `with substantially higher success rates than the baselines` | 单复数与冠词 |
| A4 | `It completed ...`；`We validated ...` | `It completes ...`；`We validate ...` | 可选的时态统一；原过去时本身不是语法错误 |

用户明确回复“第一批全部通过”。A1–A4 已应用于摘要，包括 A4 的现在时统一。

## 第 2 批：引言（用户通过，已应用）

用户回复“通过”，本批 I1–I6 已按审核稿应用。未将“Furthermore, we developed”改为现在时，因为本批未提出该项。

### I1：引用前多余句号

位置：[root.tex:95](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:95)。

```tex
% 原文
skills.~\citep{fu2024humanplus,luo2025sonic}.
framework.~\citep{ding2026unilavira,cheng2025navila}.

% 建议
skills~\citep{fu2024humanplus,luo2025sonic}.
framework~\citep{ding2026unilavira,cheng2025navila}.
```

理由：将引用放在句末标点之前，不改引用内容。

### I2：间接疑问句的句末标点

位置：[root.tex:97](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:97)。

原文：

> The research question is whether we can develop a universal approach to address the pervasive discrepancy in observation distributions, thereby seamlessly bridging navigation and manipulation skills?

建议：

> The research question is whether we can develop a universal approach to address the pervasive discrepancy in observation distributions, thereby seamlessly bridging navigation and manipulation skills.

理由：这是由 `whether` 引导的间接疑问句；本项仅将问号改为句号。

### I3：控制器的主谓一致

位置：[root.tex:99](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:99)。

原文：`the controller  perform feedback control`。

建议：`the controller performs feedback control`。

理由：单数主语搭配 `performs`，同时移除多余空格。

### I4：LaViRA 名称

位置：[root.tex:101](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:101)。

原文：`Drawing inspiration from Lavira`。

建议：`Drawing inspiration from LaViRA`。

理由：统一文中方法名称的大小写。

### I5：双 Agent 职责的句子结构

位置：[root.tex:101](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:101)。

原文：

> The high-level language agent analyze the instruction, history, and current observation to produce a overall strategy, such as direction to head, whether skill to use or managing TODO list. While a high-frequency vision agent handles low-level tasks including grounding and state monitoring.

建议：

> The high-level language agent analyzes the instruction, history, and current observation to determine an overall strategy, including which direction to take, which skill to use, and how to manage the TODO list. A high-frequency vision agent handles low-level tasks, including grounding and state monitoring.

理由：修正 `analyze`、`a overall`、`whether skill`；将列表整理为平行结构；去掉造成残句的 `While`。此建议保留原来的职责与频率描述，不新增机制解释。

### I6：贡献条目的主谓一致

位置：[root.tex:111](/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex:111)。

原文：`an agentic mobile-manipulation system which coordinate navigation, alignment, and humanoid manipulation`。

建议：`an agentic mobile-manipulation system that coordinates navigation, alignment, and humanoid manipulation`。

理由：修正单数主语与动词的搭配。

## 第 3 批：Related Work 前两段（用户统一采用，已应用）

用户回复“统一”，按本批 R1–R11 统一采用建议写法处理，已应用。

位置：root.tex 的 Related Work 标题与前两段。第二批合并一句跨行文本后，原行号向前移动一行。本批仅整理语言，不重新验证或加强对相关工作的技术论断。

| 编号 | 原文 | 建议 | 理由 |
|---|---|---|---|
| R1 | `Related Works` | `Related Work` | 可选：采用论文中惯用的章节名称 |
| R2 | `Humanoid loco-manipulation is to coordinately control a humanoid robot to move and interact with environment.` | `Humanoid loco-manipulation involves coordinated control of a humanoid robot to move and interact with the environment.` | 修正搭配并补冠词 |
| R3 | `general motion controller like SONIC, BFM-Zero scales` | `general motion controllers such as SONIC and BFM-Zero scale` | 单复数与并列结构 |
| R4 | `Meanwhile, Humanoid VLA have realized loco-manipulation skill training and deployment through multiple paradigms.` | `Meanwhile, humanoid VLA models have enabled the training and deployment of loco-manipulation skills through multiple paradigms.` | 补出 models，调整搭配与大小写 |
| R5 | `WholeBodyVLA use ... executed by RL controller` | `WholeBodyVLA uses ... executed by an RL controller` | 主谓一致与冠词 |
| R6 | `a space consist of` | `a space consisting of` | 修正分词结构 |
| R7 | `root-motion, and co-trained with heterogeneous data` | `root motion and is co-trained with heterogeneous data` | 补被动语态助动词，统一名词写法 |
| R8 | `execute moving, exploration, and object search tasks` | `perform locomotion, exploration, and object-search tasks` | 整理任务名称的并列搭配 |
| R9 | `generation of moving commands` | `generation of motion commands` | 更自然的名词搭配 |
| R10 | `instructions.~\citep{cheng2025navila}.` | `instructions~\citep{cheng2025navila}.` | 去掉引用前多余句号 |
| R11 | `LaViRA and Uni-LaViRA utilizes ... two  differently-sized models ... groundin and outputs` | `LaViRA and Uni-LaViRA utilize ... two models of different sizes ... grounding and output` | 修正复数主语的动词、拼写、多余空格和表达 |

## 第 4 批：Related Work 后两段（部分通过，已按审核结果应用）

用户当时明确要求：“除了R15,R17,都可以,然后保留任务不相关的这个表述先”。已应用 R12、R13、R14、R16、R18、R19、R20。R15、R17 仍保留；用户随后在 overclaim 审核中撤销了 `task-irrelevant` 的保留要求，见下文更新。

位置：当前 root.tex:120–123；方法名称以仓库现有 `references.bib` 中 `chen2025owmm` 的标题为依据。

| 编号 | 原文 | 建议 | 理由 |
|---|---|---|---|
| R12 | `extend single policy to mobile ma-` 换行 `nipulation systems` | `extend a single policy to mobile manipulation systems` | 补冠词，去掉导致 PDF 出现 `ma- nipulation` 的手工断词 |
| R13 | `WMM-Agent` | `OWMM-Agent` | 与对应参考文献标题统一 |
| R14 | `MoMaStage ... and trigger replanning` | `MoMaStage ... and triggers replanning` | 单数主语的动词一致 |
| R15 | `VLA, navigation and align`；`close-loop control`；`features of above works` | `VLA, navigation, and alignment`；`closed-loop control`；`features of the above approaches` | 并列结构、拼写与冠词 |
| R16 | `Handoff failure is a key problem lies in mobile-manipulation tasks.` | `Handoff failure is a key problem in mobile-manipulation tasks.` | 去掉多余谓语 |
| R17 | `through trainnning, annotation or zero-shot ways` | `through training, annotation, or zero-shot methods` | 拼写与搭配 |
| R18 | `a trainned policy` | `a trained policy` | 拼写 |
| R19 | `HERMES use a environment-specified pnp method` | `HERMES uses an environment-specific PnP method` | 主谓一致、冠词与术语大小写 |
| R20 | `geometric-aware feedback control`；`while perform well in humanoid robot platform` | `geometry-aware feedback control`；`while performing well on a humanoid robot platform` | 词形、分词结构和介词冠词 |

### 语义项：用户已撤销 task-irrelevant 的保留要求

用户最新指示：“task irrelevant不需要保留了”。已将 Related Work 中该句替换为：`Our geometric handoff module uses task-conditioned references and closed-loop geometric feedback, without requiring additional transition-policy training.` 该修改明确任务条件参考和交接模块的训练范围。R15、R17 的保留要求不受影响；系统泛化性审核 G1–G12 尚待批准。

## 第 5 批：System Overview 与 LA/VA 说明（用户通过，已应用）

用户回复“通过”。M1–M8 已应用；公式、行内数学符号以及此前指定保留的 Related Work 表述保持不变。

### M1：System Overview 首段

原文：

> We introduce a framework with dual agents and a unified skill interface to decompose planning, grounding and action execution. Figure~\ref{fig:architecture} illustrate the overall pipeline, a Language Agent(LA) is designed to call skill, making plan and managing memory, simple vision tasks like grounding and postcheck are finished by a smaller Vision Agent(VA), and finally, we use a unified skill interface to execution different skills like manipulation, navigation and align. To seamless finish such diverse tasks in one humanoid while avoid collapse, we use a unified humanoid locomotion layer to control robot motor.

建议：

> We introduce a framework with dual agents and a unified skill interface to decompose planning, grounding, and action execution. Figure~\ref{fig:architecture} illustrates the overall pipeline. A Language Agent (LA) invokes skills, makes plans, and manages memory, while a smaller Vision Agent (VA) handles simple visual tasks such as grounding and post-execution checks. A unified skill interface executes manipulation, navigation, and alignment skills. To execute these diverse tasks seamlessly on a single humanoid robot while avoiding falls, we use a unified humanoid locomotion layer to control the robot's motors.

理由：拆开逗号连接的长句，修正主谓、并列结构、词形和空格；将 `avoid collapse` 表达为“避免跌倒”（`avoiding falls`），具体措辞供用户审核。

### M2–M8：LA/VA 说明

| 编号 | 原文 | 建议 | 理由 |
|---|---|---|---|
| M2 | `Language Agents.`；`Vision Agents.` | `Language Agent.`；`Vision Agent.` | 标题与单个 LA、VA 对应 |
| M3 | `LA receive language instruction`；`global observation ... consist of five pictures`；`including last action result`；`most five history result picture` | `LA receives a language instruction`；`a global observation ... consisting of five images`；`including the last action result`；`at most five historical result images` | 主谓、冠词、分词和单复数；保留原数学符号 |
| M4 | `LA performs two task: Select a language skill ... and updating a task list` | `LA performs two tasks: selecting a language skill ... and updating a task list` | 复数与并列结构 |
| M5 | `Language action ..., is a structured JSON reply consist of state estimate, reasoning, which skill to use, predicted results and args.` | `The language action ... is a structured JSON response containing a state estimate, reasoning, the selected skill, predicted results, and arguments.` | 去除主谓间逗号，修正分词与列表结构；保留原数学符号 |
| M6 | `Task list ... is a simple text memory.`；`Appendiex` | `The task list ... serves as a simple text memory.`；`Appendix` | 冠词、搭配与拼写 |
| M7 | `VA have two main tasks: First, it is responsible to extending` | `VA has two main tasks. First, it is responsible for extending` | 主谓、标点与介词 |
| M8 | `Another task is to post-check after skill is done. In this situation, VA receive the result observation of action ... and the chosen language action ... . By checking the predicted result and actual result observation, it gives a task transition` | `The second task is to perform a post-execution check after a skill finishes. In this case, VA receives the action-result observation ... and the selected language action ... . By comparing the predicted result with the observed outcome, it produces a task transition` | 整理不自然的表达，保留原数学符号与状态集合 |

附记：历史图像文字中的“五张”与公式从 t 到 t-5 的计数不一致，以及 postcheck 公式输入与文字的对应问题，涉及符号含义。本批不自动改下标、公式输入或状态定义，待单独核对。

## 第 6 批：技能接口与全身控制（用户通过，已应用）

用户回复“通过”。M9–M16 已应用，原引用和数学符号保留。

验证：`git diff --check -- root.tex` 通过。首次编译在载入 OVMM 示例 PNG 时失败；该文件通过 PNG 完整性检查，未修改图片，重新编译成功生成 `/tmp/iclr-language-batch6-retry-20260926/root.pdf`。首次失败的原因尚未确认；仍有已有的 Underfull hbox 和 algorithm.sty 编码警告。

### M9：技能接口说明

原文：

> The skill interface are designed as a a deterministic orchestration layer between the agents and robot controllers. It provide three different categories of skill for LA and and integrates the $(A^{vis}_t,\mathcal{T}_t)$ to generate skill goals or change mission stage. Below is the skill categories:

建议：

> The skill interface is designed as a deterministic orchestration layer between the agents and robot controllers. It provides three categories of skills for LA and combines $(A^{vis}_t,\mathcal{T}_t)$ to generate skill goals or change the mission stage. The skill categories are as follows:

### M10–M16：技能条目和控制层

| 编号 | 原文 | 建议 | 理由 |
|---|---|---|---|
| M10 | `and skill layer construct a local navigation goal` | `and the skill layer constructs a local navigation goal` | 冠词与主谓一致 |
| M11 | `VA select a geometric reference described in Section ... from the language action ... and current observation, then the GATE align to the reference.` | `VA selects a geometric reference, as described in Section ..., based on the language action ... and the current observation; GATE then aligns the robot with that reference.` | 主谓、冠词与句子连接；保留引用与数学符号 |
| M12 | `Detailed introduction of skill state machine, harness and transitions are in Appendix ... .` | `A detailed description of the skill state machine, harness, and transitions is provided in Appendix ... .` | 冠词、主谓与搭配 |
| M13 | `Whole body control layer.`；`two exclusive modes` | `Whole-body control layer.`；`two mutually exclusive modes` | 复合形容词与更准确的互斥表达 |
| M14 | `high-level speed command`；`navigation and align stages` | `high-level speed commands`；`navigation and alignment stages` | 复数与名词搭配 |
| M15 | `whole-body motion reference are directly provided` | `whole-body motion references are directly provided` | 单复数一致 |
| M16 | `motion references space`；`More details of SONIC depoly` | `motion-reference space`；`Further details of SONIC deployment` | 搭配与拼写 |

## 第 7 批：导航与 GATE 语言修正（用户通过，已应用）

用户回复“修正”。N1–N7 已按审核方案应用；R15、R17 及 `task-irrelevant` 保留。独立修改对比保存于 `/tmp/iclr-language-batch7.diff`。

验证：替换核对和 `git diff --check -- root.tex` 均通过；Tectonic 编译成功，PDF 为 `/tmp/iclr-language-batch7-20260926/root.pdf`。编译仍有已有排版及宏包编码警告。

| 编号 | 原文 | 建议 | 理由 |
|---|---|---|---|
| N1 | `View Choose & Grounding` | `View Selection & Grounding` | 标题名词搭配 |
| N2 | `When choosing navigation skill`；`LA will output a chosen view as VA input`；`than VA execute grounding and robots turn to corresponding direction` | `When selecting a navigation skill`；`LA outputs the selected view as input to VA`；`then VA performs grounding and the robot turns toward the corresponding direction` | 冠词、主谓、介词与 then 拼写；同时把原逗号改为句号，将 then 大写为 Then，拆开两句 |
| N3 | `Given the grounding ... which is a bounding box, we select the 7$^*$7 center pixel area and performs spatial and temporal pooling to get depth ... . Than unprojecting to camera's 3D coordinate system using the intrinsic matrix ... :` | `Given a grounding bounding box ..., we select a $7\times7$ pixel region at its center and perform spatial and temporal pooling to estimate depth ... . We then unproject the center pixel into the camera's 3D coordinate system using the intrinsic matrix ... :` | 修正主谓和残句，使用乘号表示像素窗口；保留原数学变量 |
| N4 | `v_{taerget}` | `v_{target}` | 公式下标拼写修正；不改变其他公式符号 |
| N5 | `Given the target point ..., robot position ..., RGB-D input ..., navigation skill use ... to generate robot trajectory, than a MCP controller converts the trajectory into speed command ...` | `Given the target point ..., the robot position ..., and the RGB-D input ..., the navigation skill uses ... to generate a robot trajectory. An MPC controller then converts the trajectory into speed commands ...` | 冠词、主谓、并列与缩写；MPC 与附录控制器名称一致；保留已有引用 |
| N6 | `To get the actual position of ... and ... in world coordinate, a SLAM algorithm based on onboard Lidar is used ... . More details in Appendix ... .` | `To obtain the positions of ... and ... in world coordinates, we use a SLAM algorithm based on the onboard LiDAR ... . Further details are provided in Appendix ... .` | 单复数、冠词、缩写及残句；保留原变量和引用 |
| N7 | `These two processes require distinct capabilities, we therefore use ...` | `These two processes require distinct capabilities. We therefore use ...` | 拆开逗号拼接的完整句子 |

本批保留原文关于转身与 grounding 的叙述顺序，以及局部避障职责描述；这两处与其他章节的技术对应关系另列待核对，不在纯语言修改中自动调整。

## 后续待处理范围

按正文顺序继续准备并审核 Related Work、System Overview、导航与控制说明、GATE、实验说明、附录。研究结论、实验数据和技术含义的调整须单独标明，不能混入纯语法修改。
