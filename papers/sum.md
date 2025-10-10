## A Survey of LLM-based Agents: Theories, Technologies, Applications and Suggestions


1. 理论基础

早期Agent基于强化学习,认为AI智能体可以基于简单的启发式驱动策略函数行动。然而，基于强化学习的智能体可能面临一些障碍[3]，如训练时间长、采样效率低和学习过程不稳定等。

由于具备卓越的多模态理解与生成能力、无与伦比的知识获取与推理能力，以及大语言模型（LLMs）的灵活性和可扩展性，人工智能Agent将LLMs作为核心大脑，试图实现人类水平的感知、认知和行为[5]。

多模态感知实现自主感知能力,复杂的规划实现自发工作能力,可行的具身化或工具利用实现反应能力,有效的多元化记忆实现交互能力。

- 模态编码器（Modality Encoder）：输入多模态原始数据,编码出其他模态的特征向量。
- 模态连接器（Modality Connector）：将模态数据和文本编码数据融合,实现对齐,输出对齐后的模态特征。
- LLM主干（LLM Backbone）：
- 输出投影器（Output Projector）
- 模态生成器（Modality Generator）

？？？不是一个LLM的主干,像是文生图的workflow

任务目标g、环境e、提示集p及语言模型总体参数Θ后，任务分解可形式化表示为[8]：g₀,g₁,...,gₙ = decp(g,e,p,Θ)

其中"decp"表示分解操作，g₀,g₁,...,gₙ代表子目标。Agent应支持通过CoT实现的单路径和多路径推理能力。

根据内外双重对齐来进行多路径推理的选择：外部对齐需要将人类意图或预期目标转化为基于大语言模型智能体的训练目标，通常采用包含监督微调、奖励建模和策略优化的RLHF方法；内部对齐则要求规划过程确保内部优化目标与智能体训练目标保持一致，具体强调通过安全评估、可解释性验证和人类价值观检验来保障规划的对齐性。

然后就是RAG通过外部知识库增强记忆与知识调用能力。

以及例如API调用、代码解释器乃至具身化工具的工具使用能力。

论文首先回顾了支撑 LLM-based Agent 的几大核心理论：

2. 关键技术（
论文将 LLM-based Agent 的技术架构归纳为四大模块：

感知（Perception）：处理文本、图像、音频、视频等多模态输入，如 BLIP-2、MiniGPT-4、AudioGPT 等。

规划（Planning）：包括任务分解、单/多路径推理（如 CoT、ToT、GoT）、反思机制（如 Reflexion、CRITIC）等，提升决策合理性。

记忆（Memory）：涵盖交互内记忆、跨交互记忆和外部知识。

- 内部交互记忆指单次交互内的历史信息
- 跨交互记忆指跨越多轮交互积累的长期历史信息
- 或是通过RAG技术获取高质量外部知识

行动（Action）：进行工具理解、工具调用、工具整合的能力。

3. 应用与评估

LLM-based Agent 已广泛应用于多个领域：

自然科学：数学（ToRA）、化学（ChemCrow）、生物学（BSDG）；

社会科学：经济学（Alpha-GPT）、法律（LJP-Agent）、心理学（Replika-MWS）；

工程领域：代码生成（GPT-Engineer、AutoGen）、游戏（Voyager、GITM）、工业规划（LLM-Planner）；

评估基准：如 AgentBench（多环境评估）、ToolLLM（工具使用）、SafetyBench（安全性）、AlignBench（中文对齐）等。

4. 挑战与建议

作者指出当前 LLM-based Agent 面临的关键挑战，并提出四点建议：

突破内在限制：如幻觉、长上下文处理、多模态推理等；

推动规模化多智能体系统：实现动态调度与高效协作；

强化可控的 AI 对齐：确保遵守法律、伦理与人类价值观；

构建统一综合的评估体系：当前评估分散，亟需标准化平台。

## Survey on evaluation of llm Agent

