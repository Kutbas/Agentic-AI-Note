# Agentic AI

## 序言

**Agentic AI：Welcome!** 欢迎来到智能代理（Agentic AI）的世界。

近年来，在构建基于大模型的应用程序时，一种全新的范式正在迅速崛起。最初，“Agentic（代理化/智能体化）”这个词被创造出来，是为了精准描述这种让 AI 自主规划、调用工具并执行复杂任务的发展趋势。

然而，技术的发展总是伴随着市场的狂热。各类营销人员迅速捕捉到了这个词，把它当成万能的标签贴在各种产品上，导致关于 Agentic AI 的炒作一度直冲云霄。但好消息是，如果我们拨开这些营销的迷雾，会发现**利用 Agentic AI 构建的真正有价值、有实用意义的应用程序数量，正在以惊人的速度增长。**

**1. Agentic Workflows 的强大应用场景**

在过去，单次对话（Zero-shot prompting）往往只能解决简单问题；而今天，代理工作流（Agentic workflows）正在重新定义 AI 的能力边界。许多过去根本无法想象的复杂项目，如今在代理工作流的加持下已成为现实。

从功能落地的角度来看，Agentic AI 已经在以下高价值场景中大展拳脚：
*   **智能客服系统**：不再是死板的问答机器人，而是能自主查询订单、调用退款接口的真正 Support Agents。
*   **深度研究与分析**：能够自主搜集海量资料、筛选信息，并撰写具有深度的专业研究报告。
*   **复杂文档处理**：高效解析和处理冗长且专业性极强的法律卷宗或商业合同。
*   **医疗辅助诊断**：通过分析患者的输入症状和病史，为医生提供可靠的诊断建议参考。


**2. 核心差异化能力：评估与错误分析 (Evals & Error Analysis)**

随着 Agentic AI 的普及，一个关键问题浮出水面：**到底是什么决定了一个开发者能否构建出优秀的代理应用？**

行业经验表明，真正精通 Agentic workflows 的高手与普通开发者之间最大的区别，不在于谁写的 Prompt（提示词）更花哨，而在于**是否具备驱动严谨开发流程的能力**。

这种严谨的开发流程，其核心功能点可以概括为两个词：**Evals（评估）** 与 **Error Analysis（错误分析）**。在构建 Agent 时，AI 的输出具有一定的不确定性，因此我们需要建立一套自动化的评估机制来量化 Agent 的表现，并在其出错时进行精准的根因分析。

为了更直观地理解这一过程，我们可以用下面这段 Python 伪代码来演示一个标准 Agent 的开发与迭代闭环：

```python
# 伪代码：Agentic AI 的严谨开发流程 (Evals & Error Analysis)

def agentic_development_loop(agent, dataset):
    """
    通过持续的评估和错误分析来优化 Agent
    """
    for test_case in dataset:
        # 1. 执行代理工作流
        agent_trajectory = agent.run_workflow(test_case.input)
        
        # 2. Evals（评估）：量化 Agent 在工具调用、逻辑推理和最终输出上的表现
        evaluation_score = run_evals(agent_trajectory, test_case.expected_outcome)
        
        # 3. Error Analysis（错误分析）：如果表现不达标，进入诊断流程
        if evaluation_score < PASSING_THRESHOLD:
            # 追踪是哪一步出错：是理解错误、工具调用失败，还是陷入死循环？
            error_logs = diagnose_agent_reasoning(agent_trajectory)
            
            # 迭代优化：修改系统提示词、增加上下文或修复工具函数
            refine_agent_configuration(agent, error_logs)
            
    return "Agent optimized and ready for production!"
```

如上所示，构建 Agentic AI 并非一蹴而就，而是一个基于数据和反馈不断自我修正的工程化过程。

**3. 拥抱 AI 时代的最核心技能**

掌握如何利用代理工作流构建应用程序，毫无疑问是当今 AI 领域最重要、也最有价值的技能之一。

无论你是希望在职场中获得更广阔的职业发展机会，还是单纯地想凭借一己之力开发出令人惊叹的软件产品，理解并熟练运用 Agentic AI 都将为你打开一扇充满无限可能的大门。

告别概念炒作，回归技术本质。接下来，我们将进一步深入探讨 Agentic workflows 的底层运作机制，敬请期待。

### Agentic Workflows 入门

在探讨大语言模型（LLM）的应用时，我们必须先理清一个核心概念：究竟什么是 Agentic AI（代理化人工智能）？为什么 Agentic Workflows（代理工作流）会拥有如此强大的能力？

要理解这一点，我们需要对比一下传统的大模型使用方式与代理工作流之间的本质区别。

**1. 传统 Non-agentic 模式：带着镣铐跳舞的“零样本（Zero-shot）”生成**

目前，大多数人使用 LLM 的方式是非常直接的：给出一个 Prompt（提示词），比如“请以 X 为主题写一篇文章”，然后等待模型输出结果。

从底层运行逻辑来看，这种 **Non-agentic workflow（非代理工作流）** 是一个完全线性的过程（`Start -> Finish`）。这就像是要求一个人（或 AI）在键盘上写文章，**从第一个字敲到最后一个字，中间绝对不允许使用“退格键（Backspace）”进行任何修改**。

人类在被强制要求以这种绝对线性的方式写作时，是无法产出最佳作品的，AI 模型同样如此。尽管在如此苛刻的限制下，当前的 LLM 依然表现得令人惊讶，但这远远没有释放出它们的真正潜力。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411180931199.png)

**2. 破局之道：Agentic Workflows 的多步迭代哲学**

与之形成鲜明对比的是 **Agentic Workflow（代理工作流）**。

**代理工作流的核心定义是：一个基于 LLM 的应用程序，通过执行多个拆解后的步骤来完成一项复杂任务的过程。**

它不再要求模型一次性生成最终结果，而是模拟人类的真实工作模式，引入了思考、调研、修改的循环（`Revise -> Thinking/research -> Revise`）。虽然这种迭代过程在计算和时间上开销更大，但交付的成果质量却有质的飞跃。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411181226297.png)

我们以“撰写一篇文章”为例，来看看一个完整的代理工作流是如何运转的，以及各组件是如何协作的。为了更清晰地展示其内部逻辑，我们可以用一段编排（Orchestration）伪代码来示意：

```python
# 伪代码：一个典型的 Agentic Workflow (文章撰写/深度研究)

def agentic_research_and_writing_workflow(topic):
    # 步骤 1：LLM 规划大纲 (Planning)
    outline = planner_agent.generate_outline(topic)
    
    # 步骤 2：LLM 决定搜索词，并调用外部工具 (Tool Use / Web Search API)
    search_queries = search_agent.formulate_queries(outline)
    raw_web_data = web_search_api.execute(search_queries)
    
    # 步骤 3：综合信息并撰写初稿 (Drafting)
    first_draft = writer_agent.write_draft(outline, raw_web_data)
    
    # 步骤 4：多 Agent 协作，由另一个 LLM 进行审稿反思 (Reflection/Critique)
    critique_feedback = editor_agent.review(first_draft)
    
    # 步骤 5：Human-in-the-loop (可选：人工介入审核关键事实)
    if requires_human_review(topic):
        human_feedback = request_human_review(first_draft, critique_feedback)
        critique_feedback.append(human_feedback)
    
    # 步骤 6：根据反馈进行修订 (Revision)
    final_essay = writer_agent.revise(first_draft, critique_feedback)
    
    return final_essay
```

正如代码所示，黑板上原本抽象的 `(LLM + LLM) + Web search -> LLM -> request human review` 流程，在工程实现中被转化为了一系列职责明确的节点。一个大模型被赋予了不同的“人设”或功能（如 Planner, Searcher, Editor），并通过调用搜索引擎等外部工具，最终完成了一次高质量的内容输出。

**3. 核心能力构建：任务拆解与重组**

在开发代理应用时，最具挑战性同时也最重要的技能，就是**如何将一个庞大且复杂的任务，拆解为代理工作流可以一步步执行的微小组件**。

以构建一个“深度研究 Agent”为例。假设用户输入了一个极具挑战性的课题：“*如何创办一家新的火箭公司来与 SpaceX 竞争？*”

如果仅靠传统的 Zero-shot 提示词，模型大概率只能基于训练数据生成一些泛泛而谈的废话。但是，如果交由深度研究 Agent 处理，系统会先进行背景调研规划，调用搜索引擎抓取最新的航空航天产业网页，对信息进行清洗、综合与排序；接着生成大纲，并让“编辑 Agent”审查逻辑的连贯性；最终，系统会输出一份包含引言、行业背景、可行性分析等模块的详尽 Markdown 格式研究报告。

通过多信息源的抓取和深度的逻辑反思，这份报告的深度和思想性将彻底碾压传统的单次生成结果。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411181340824.png)

**4. 走向现实应用与自主性探讨**

这种将复杂任务拆解并由 Agent 分步执行的架构，具有极高的商业价值。目前，专门定制的研究型 Agents 已经被广泛应用于各大垂直领域，例如：
*   **法律合规**：自主检索冲突条款并出具法务审查意见。
*   **医疗健康**：聚合最新医学文献，辅助特定疾病的诊疗研究。
*   **商业产品**：进行深度的竞品分析与市场前景调研。


当我们惊叹于这些由多个 LLM 节点和工具链交织而成的复杂工作流时，一个经常被讨论的问题自然浮现：**这些 AI 代理到底有多高的自主权（Autonomy）？**

刚才展示的火箭公司调研案例，是一个相对复杂且高度自主的工作流。但实际上，还有许多设计更简单、自主性稍弱但同样极具价值的工作流模式。在接下来的内容中，我们将引入一个全新的框架，深入探讨 Agentic Workflows 的不同“自主程度”，以及如何根据具体的应用场景，评估开发不同级别 Agent 的难度与策略。

### Agent 的自主程度

在探讨 AI 代理的演进时，我们经常会陷入一个误区。几年前，AI 社区内曾爆发过一场激烈的争论：**到底什么样的系统才配得上“Agent（智能体）”这个称号？**

有人发表论文宣称自己构建了一个 Agent，紧接着就会有反对声音跳出来说：“不，你的系统不够自主，那根本不是真正的 Agent。” 这种非黑即白的二元对立（“IT'S AN AGENT! NO, IT'S NOT!”），在很大程度上消耗了开发者的精力。

正如吴恩达（Andrew Ng）等行业先驱在 2024 年提出并倡导的那样：**我们不应再纠结于将某个系统归类为“是”或“不是”Agent，而应该将“Agentic（代理化）”视为一个形容词。** 我们需要承认，不同的系统具备不同程度的“代理化”特征。停止无意义的争论，将精力集中在构建实际的系统上，这才是推动技术落地的正确路径。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411182256632.png)

**1. 自主性光谱：从固定流水线到动态决策**

为了更直观地理解 Agentic 系统的不同自主程度（Degrees of autonomy），我们可以通过一个具体的任务来对比——“**撰写一篇关于黑洞的文章**”。

在架构设计中，我们通常将系统抽象为三种核心节点：**输入节点**（用户给定的文档或指令）、**推理节点**（LLM 的思考与生成）以及**动作节点**（调用外部软件/API，如执行 Web 搜索）。系统在这些节点之间的路由方式，决定了它的自主性级别。

**(1) 低自主性 (Less Autonomous)：预设的确定性工作流**

在低自主性系统中，工作流的每一个步骤都是由人类工程师硬编码（Hard-coded）提前定义好的。LLM 就像流水线上的工人，只负责执行特定环节的文本生成任务。

其工作流通常是线性的：
*接收请求 -> LLM（生成搜索词） -> 强制执行网络搜索 -> 获取网页内容 -> LLM（根据内容写文章）*。

我们可以用以下伪代码来表示这种强控制流的架构：

```python
# 伪代码：低自主性 Agentic Workflow (强控制、固定步骤)

def less_autonomous_agent(topic):
    # 步骤 1：固定调用 LLM 生成搜索词
    search_queries = llm.generate("Write search queries for: " + topic)
    
    # 步骤 2：硬编码的工具调用，没有选择余地
    web_content = web_search_tool.execute(search_queries)
    
    # 步骤 3：固定调用 LLM 生成最终文章
    essay = llm.generate("Write an essay using this context: " + web_content)
    
    return essay
```
在这种模式下，**系统的自主性仅仅体现在 LLM 生成的文本内容上**。它的优点是极其稳定、易于控制，目前工业界大量产生高商业价值的 AI 应用，往往都采用了这种低自主性但高确定性的架构。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411182331190.png)

**(2) 高自主性 (Highly Autonomous)：LLM 掌舵的动态路由**

当我们赋予 LLM 更多的决策权时，系统就迈向了高自主性。同样是写关于黑洞的文章，高自主性 Agent 不再遵循死板的流水线。

它会自主评估需求，并在工具箱（Tools）中进行选择：是进行普通的 Web 搜索，还是去检索最新的新闻，亦或是去 arXiv 查找学术论文？找到 PDF 后，它还会自主决定是否需要调用 `pdf_to_text` 工具进行解析。如果写完初稿觉得不够好，它甚至会自主发起新一轮的搜索和修改（Reflect and improve draft）。

```python
# 伪代码：高自主性 Agentic Workflow (动态决策、自主循环)

def highly_autonomous_agent(topic):
    # 初始化状态与可用工具库
    agent_state = State(goal=topic)
    tools = [WebSearch(), NewsAPI(), ArxivSearch(), PDFParser()]
    
    # LLM 完全掌控执行循环
    while not agent_state.is_completed:
        # LLM 自主决定下一步动作和所需调用的工具
        next_action, selected_tool = llm.plan_next_step(agent_state, tools)
        
        if next_action == "USE_TOOL":
            result = selected_tool.execute(agent_state.context)
            agent_state.update(result)
            
        elif next_action == "REFLECT_AND_REVISE":
            draft = llm.generate_draft(agent_state.context)
            critique = llm.reflect(draft)
            agent_state.update(critique) # 根据反思更新状态，可能触发下一轮搜索
            
        elif next_action == "FINISH":
            agent_state.is_completed = True
            
    return agent_state.final_output
```

在这种高度自主的模式下，**确切的执行步骤是由 LLM 动态决定的，而非程序员提前规划**。甚至在某些前沿研究中，这类 Agent 还具备即时编写代码来创建全新工具（Create new tools on the fly）的能力。

**3. 总结：如何选择适合的自主性级别？**

纵观 Agentic AI 的自主性光谱，我们可以将其粗略划分为三个层级：

*   **低自主性（Less autonomous）**：所有步骤预设，工具硬编码。可控性极强，适合对稳定性要求极高的商业落地场景。
*   **半自主性（Semi-autonomous）**：介于两者之间。Agent 可以在预定义的工具池中进行选择和有限的路径决策。
*   **高自主性（Highly autonomous）**：Agent 自主决定绝大部分执行路径，甚至动态创造工具。其上限极高，但不可控性和不可预测性也随之增加，目前仍是学术界和前沿研发的活跃探索领域。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411182940631.png)


掌握 Agentic Workflows 的关键，并不在于一味追求最高级别的自主性，而在于**根据实际的业务需求和容错率，在自主性光谱上找到最合适的那个点**。低端光谱中蕴藏着无数立竿见影的商业机会，而高端光谱则代表着通向通用人工智能（AGI）的未来脉络。

理解了自主性的分级框架后，在接下来的探讨中，我们将进一步深挖这些 Agentic 工作流底层的设计模式，揭示它们如何赋予基础模型前所未有的强大能力。

### Agentic AI 的优势

在深入了解了代理工作流（Agentic Workflows）的运行机制后，一个关键的问题随之而来：**既然代理工作流需要执行多次调用，耗时更长，为什么我们还要大费周章地构建它？**

答案在于，Agentic 架构赋予了我们完成过去根本无法触及的复杂任务的能力。除了显著打破基础大语言模型（LLM）的性能天花板之外，它在并发处理和架构解耦上同样展现出了降维打击般的优势。

**1. 性能跃升：Agentic 包装的收益甚至大于模型代际升级**

为了直观地衡量 Agentic 工作流的威力，我们可以参考代码生成领域的权威评测基准——**HumanEval**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411184202025.png)

在传统的非代理（Non-agentic / Zero-shot）模式下，仅仅通过直接输入 Prompt 让模型一次性输出代码：
*   **GPT-3.5**（早期 ChatGPT 的基础模型）的准确率仅为 **48%**。
*   **GPT-4** 作为更强大的基座模型，其准确率跃升至 **67%**。


从 48% 到 67%，这是模型本身跨代升级带来的巨大进步。然而，真正令人震撼的发现是：**当我们将较弱的 GPT-3.5 嵌入到一个包含自我反思（Reflection）和迭代优化的 Agentic 工作流中时，它的表现竟然能够大幅超越非代理模式下的 GPT-4！**

学术界和工业界已经涌现出大量优秀的 Agentic 框架（如 Reflexion、Language Agent Tree Search、MetaGPT、AgentCoder 等）。它们的核心逻辑是：让模型先写代码，接着运行测试用例，如果报错，再让模型根据报错信息“反思”并修改代码。

我们可以用一段 Python 伪代码来揭示这种性能跃升的底层逻辑：

```python
# 伪代码：以 Reflexion 模式包装的 Agentic 编码工作流

def agentic_coding_workflow(prompt, max_retries=3):
    # 首次尝试生成代码
    code_draft = base_model.generate_code(prompt)
    
    for attempt in range(max_retries):
        # 执行代码并捕获结果
        test_result = execution_environment.run(code_draft)
        
        if test_result.is_success:
            return code_draft # 成功则直接返回
            
        # 核心魔法：让模型进行反思 (Reflection)
        critique = base_model.analyze_errors(code_draft, test_result.error_logs)
        
        # 根据反思结果修改代码
        code_draft = base_model.improve_code(code_draft, critique)
        
    return code_draft
```

这意味着，哪怕是使用当今最顶级的 LLM，套用 Agentic 工作流依然能带来巨大的性能红利。**构建良好的代理流程带来的收益，往往比单纯等待下一代大模型发布还要可观。**

**2. 并行化提速 (Parallelization)：超越人类直线的处理极限**

虽然 Agentic 工作流由于多次调用 LLM，整体耗时比单次生成要长，但如果将其与“人类完成同等复杂任务”的时间相比，Agent 会凭借其**并行处理能力**形成压倒性优势。

以“撰写关于黑洞的深度文章”为例。如果是一个人类研究员，他需要先想出几个搜索词，然后在搜索引擎里一个个搜索，再把排名前 9 的网页逐一打开、按顺序阅读，最后提炼总结。这是一个纯粹的串行（Sequential）过程。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411184606390.png)

但在 Agentic 架构中，我们可以利用异步 I/O 大幅压缩时间：

```python
# 伪代码：Agentic 工作流中的异步并发处理
import asyncio

async def parallel_research_workflow(topic):
    # 1. LLM 并行生成 3 个不同的搜索方向
    search_queries = await llm.generate_queries(topic, count=3)
    
    # 2. 并发执行 3 次独立的网络搜索
    search_tasks = [web_search_api.search(q) for q in search_queries]
    search_results = await asyncio.gather(*search_tasks)
    
    # 3. 提取前 9 个高价值 URL，并同时发起 9 个网页的并发下载与解析
    urls = extract_top_urls(search_results, total=9)
    fetch_tasks = [web_fetcher.download_and_parse(url) for url in urls]
    pages_content = await asyncio.gather(*fetch_tasks)
    
    # 4. 将海量内容一次性喂给 LLM 撰写文章
    essay = await llm.write_essay(pages_content)
    return essay
```

通过并发执行（Parallelization），原本需要人类耗费数小时逐字阅读和抓取的资料，Agent 可以在几秒钟内通过开启多个并行线程（如 3 路并发搜索 -> 9 路并发下载）瞬间完成。这种能力让 AI 在处理重度数据搜集型任务时，拥有了超越人类物理极限的速度。

**3. 极致的模块化 (Modularity)：组件的无缝插拔与替换**

构建 Agentic 工作流的第三大乐趣，在于其高度解耦的**模块化设计**。一个成熟的 Agent 系统就像是乐高积木，你可以随时添加、更新或替换其中的单个组件，而无需重构整个系统。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260411184945509.png)

*   **工具链的灵活替换**：比如在搜索节点，如果你觉得传统的 Google 搜索不够聚焦，你可以无缝将其替换为专为 LLM 设计的搜索引擎（如 Tavily、DuckDuckGo 或 u.com）。你甚至可以随时插入一个新的“新闻搜索 API”，让 Agent 专门去获取黑洞科学的最新突破。
*   **模型路由（Model Routing）**：在同一个工作流中，你不必从头到尾使用同一个模型。例如，在“网页内容提取”这种简单任务上，你可以调用便宜且快速的小模型；而在最后一步“反思与润色”时，再切换到推理能力最强、成本较高的旗舰大模型。这种模块化组合，能够完美平衡性能与 API 成本。


**4. 总结**

概括而言，代理工作流之所以成为当前 AI 开发的绝对主流，归功于它所带来的三大核心优势：

1.  **指数级的性能提升**：通过反思和迭代，让小模型超越大模型，让大模型突破极限。
2.  **并发提速**：打破人类的线性工作瓶颈，实现海量信息的同时处理。
3.  **高度的模块化**：工具可升级，模型可插拔，系统具备极强的生命力和扩展性。


在掌握了这些核心组件和理论优势之后，我们的技术视野已被彻底打开。接下来，我们将正式步入实战案例环节，去看看业界目前都在使用这些 Agentic 组件构建哪些令人惊叹的实际应用。

### Agentic AI 应用示例

在深入了解了 Agentic Workflows（代理工作流）的底层机制与核心优势之后，我们有必要将目光投向现实世界。目前，业界究竟在使用这些技术构建怎样的应用？不同应用之间的开发难度差异又在哪里？

通过拆解几个典型的业务场景，我们可以清晰地看到 Agentic AI 的能力边界与演进方向。

**1. 基础落地：清晰且确定的“标准作业程序 (SOP)”**

在企业中，存在大量具有明确步骤的重复性流程，这类任务往往是 Agentic 工作流最容易落地，且能最快产出商业价值的领域。

**场景 A：自动化发票处理 (Invoice Processing)**
假设系统收到一份 PDF 格式的发票（例如 *TechFlow Solutions LLC, 包含金额 $3,000 和到期日 2025年8月20日*）。过去，这需要财务人员手动录入。而在代理工作流中，系统会执行一个极度确定的管线：
首先调用 `PDF-to-text API` 将文件转为 Markdown 文本；接着，LLM 会判断这是否真是一张发票；确认无误后，LLM 会精准提取“开票方、地址、金额、日期”四个必填字段，最后自主调用 `Update Database` 工具，将记录写入底层数据库。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412123425455.png)

**场景 B：售后邮件回复与 Human-in-the-loop（人类反馈介入）**
再比如处理一封客户抱怨“发错货”的邮件（*本该发蓝色搅拌机，却收到了红色烤面包机*）。Agentic 工作流可以完美集成人工审核机制。

我们可以用一段伪代码来展示这种包含人工介入（Human-in-the-loop）的确切工作流设计：

```python
# 伪代码：集成数据库查询与人工审核的客服邮件 Agent

def process_customer_email(email_text):
    # 步骤 1：LLM 提取关键实体（如：订单号 #8847，问题：发错货）
    extracted_info = llm.extract_entities(email_text)
    
    # 步骤 2：调用工具查询企业数据库，验证订单真实性
    order_record = orders_database.query(extracted_info.order_id)
    
    # 步骤 3：LLM 基于数据库记录，起草安抚与解决策略的回复
    draft_response = llm.draft_email(email_text, order_record)
    
    # 步骤 4：调用审查工具，将草稿推入人工审核队列
    # Agent 暂停执行，等待人类确认后才会真正发送
    review_tool.request_human_review(draft_response, target_status="pending_approval")
    
    return "Draft submitted for human review."
```

在这两个案例中，**所需执行的步骤在开发前就已经完全确定**。Agent 只需要按部就班地提取信息、调用指定工具即可。这属于技术实现上相对“简单（Easier）”的范畴。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412123851866.png)

**2. 进阶挑战：需要“动态规划”的复杂推理场景**

当我们将客户服务代理的应用范围扩大，允许用户提出任何开放式问题时，系统的复杂度就会呈指数级上升。

假设客户问：“*你们有黑色或蓝色的牛仔裤吗？*” 或者 “*我想退掉之前买的沙滩毛巾。*”

对于这类请求，**系统无法提前预知需要执行哪些具体步骤**。Agent 必须具备“边走边想（Plan/solve as you go）”的动态规划能力：
*   面对库存查询，Agent 需要自主决定先调用 API 查询黑色牛仔裤库存，再查询蓝色库存，最后综合结果回复用户。
*   面对退货请求，Agent 必须先验证用户的购买记录，接着查询系统退货政策（例如：是否在 30 天内？），如果符合条件，它还要连续执行两个动作：生成退货标签（Issue return packing slip）并修改数据库状态为“退货处理中（return pending）”。


这种高度依赖 LLM 现场决策和多步逻辑推理的系统，容错率更低，开发难度也显著增加。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412124242145.png)

**3. 前沿探索：多模态与视觉计算机操作 (Visual Computer Use)**

目前 Agentic AI 最激动人心、也最困难的前沿研究领域之一，是让 Agent 像人类一样直接“看”屏幕并操作浏览器界面。

以“*查询旧金山到华盛顿（DCA）的特定美联航航班余票*”为例。在高级 Agent 模式下（如 OpenAI 正在测试的特性），Agent 会被赋予浏览器的控制权：
它会打开美联航官网，通过视觉或 DOM 树理解页面结构，自主点击输入框并键入机场代码。如果美联航网站卡顿或报错，高级的 Agent 甚至能展现出惊人的应变能力——它会主动放弃当前页面，导航至 Google Flights（谷歌航班）进行交叉检索，找到匹配结果后再跳回美联航官网完成最终验证。

然而，尽管演示效果令人惊艳，这类系统目前仍面临巨大的挑战。网页加载的延迟、复杂的动态 UI 元素，都可能导致 Agent 失去对当前状态的理解。因此，视觉计算机操作目前尚无法广泛应用于容错率极低的核心关键任务（Mission-critical applications）中。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412124455406.png)

**4. 架构师视角：如何评估 Agentic 项目的落地难度？**

综合以上案例，我们在设计和立项 Agentic AI 工作流时，可以遵循一个明确的**任务难度评估光谱**：

*   **极易落地（Easier）**：
    *   业务本身有清晰的、按部就班的操作流。
    *   企业内部已经存在现成的标准作业程序（SOP）。
    *   输入输出仅限于纯文本数据（Text assets only），因为当前 LLM 对文本的解析最为稳定。
*   **极具挑战（Harder）**：
    *   执行步骤无法预知，需要 Agent 根据实时反馈进行动态规划（Plan as you go）。
    *   系统需要处理丰富的多模态输入（Multimodal），如解析音频、理解复杂的视觉图像或实时网页截屏。

    ![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412124600424.png)


**5. 总结与展望**

无论业务需求落在难度光谱的哪一端，想要成功构建 Agentic 工作流，最核心的工程化能力都在于**任务拆解（Task Decomposition）**。

面对撰写深度研究报告或全能客服助理这样庞大的目标，我们如何将其优雅地拆解为一个接一个、Agent 可以稳健执行的离散化组件？在接下来的内容中，我们将深入探讨任务拆解的底层逻辑与实践方法。

### 任务拆解与组件化思维

当我们惊叹于 Agentic AI 能够处理极其复杂的业务需求时，往往会产生一个疑问：**我们究竟该如何把现实世界中复杂、模糊的人类工作，转化为 AI 能够无缝执行的系统代码？**

答案隐藏在构建所有优质软件的黄金法则中：**任务拆解（Task Decomposition）**。

在设计 Agentic Workflow（代理工作流）时，最核心的架构能力就是将宏大的目标拆解为一个个离散的（Discrete）、可由 AI 或代码独立完成的微小步骤。

**1. 迭代式拆解：以“撰写深度长文”为例**

假设我们的目标是让系统“撰写一篇关于主题 X 的深度研究文章”。

**第一阶段：直接生成 (Direct Generation) —— 浅尝辄止**
如果我们采用非代理（Non-agentic）的暴力美学，直接用 Prompt 命令大模型生成结果，往往会发现文章浮于表面（Surface level），只涵盖了明显的常识，缺乏深度。**结论：不够好（Still not good enough）。**

**第二阶段：三步工作流 (3-Step Workflow) —— 引入外部知识**
此时，我们需要代入人类思考模式：一个优秀的研究员会怎么写？他会先列大纲，然后查资料，最后动笔。
将其转化为 Agentic 工作流，我们可以拆解为三步：
1.  **写大纲**（LLM 擅长思考结构）
2.  **网络搜索**（LLM 生成搜索词 + Web Search API）
3.  **撰写正文**（LLM 汇总信息并生成）


这是个合理的尝试。但在实际工程中，我们可能会发现生成的文章前后逻辑脱节（Disjointed）。**结论：依然不够好。**

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412125455051.png)

**第三阶段：五步工作流 (5-Step Workflow) —— 引入反思与自我纠错**
为了解决连贯性问题，我们需要对“撰写正文”这一步进行**二次拆解**。正如人类作家一样，AI 也需要“草稿-审阅-修改”的过程。
最终的拆解方案如下：

1.  **规划**：撰写大纲
2.  **检索**：网络搜索
3.  **起草**：撰写初稿
4.  **反思**：审视初稿中需要修订的部分（LLM Critique）
5.  **修订**：根据反思结果修改草稿


通过这种迭代式的拆解，原本生硬的“直接生成”，演变成了一个具有极高思想深度的智能流水线。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412125816315.png)

**2. 如何拆解业务流？**

在商业应用中，任务拆解的逻辑同样适用。每拆解出一步，我们都要问自己一个关键的工程问题：**这一步，能否被一个 LLM、一段简短的代码，或者一次 API 调用所解决？**

**案例 A：处理售后客服邮件**
输入一封充满抱怨的邮件，拆解如下：
*   **Step 1：提取关键信息** -> *纯 LLM 任务*（提取姓名、订单号 #8847 等）
*   **Step 2：查找客户记录** -> *LLM + 工具库*（LLM 生成 SQL/查询条件 -> 调用 Orders Database API）
*   **Step 3：起草并发送回复** -> *LLM + 工具库*（LLM 写邮件 -> 调用 Send Email API）

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412130443586.png)


**案例 B：发票信息结构化入库**
输入一张非结构化的发票 PDF，拆解如下：

*   **前置处理**：PDF to Text API（将图像转为文本）
*   **Step 1：寻找必须字段** -> *纯 LLM 任务*（在文本中精准定位账单方、金额等 4 个关键字段）
*   **Step 2：创建并保存记录** -> *LLM + 工具库*（调用 Update Database API）


如果某一步的答案是“否”，说明这一步依然过于宏大，我们需要像剥洋葱一样，继续对其进行向下拆解。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412130725068.png)

**3. Agent 架构师的“弹药库”：构建块 (Building Blocks)**

在进行任务拆解时，你的设计方案受限于你对**系统底层能力的认知**。构建 Agentic 工作流，本质上就是在对各类**构建块（Building blocks）**进行排列组合。

目前，我们可以调用的主要武器库包括：

1.  **大模型 (Models)**
    *   **LLMs / 多模态大模型**：系统的“大脑”，负责文本生成、逻辑推理、工具路由、信息提取。
    *   **专用 AI 模型**：负责单一维度的感知转换，如 PDF 解析、Text-to-Speech (TTS)、图像识别等。
2.  **工具集 (Tools / APIs)**
    *   **信息获取**：Web 搜索 API、实时天气接口。
    *   **动作执行**：发送邮件、修改日历、调用企业内部数据库。
    *   **知识检索 (Information Retrieval)**：连接企业私有数据的 RAG（检索增强生成）系统。
    *   **代码执行器 (Code Execution)**：允许大模型在沙盒中实时编写并运行 Python 代码，以完成数据分析或复杂数学计算。


架构师的职责，就是敏锐地观察业务流，然后像搭乐高积木一样，用这些“大脑（Models）”和“手脚（Tools）”去重构工作流。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412130856324.png)

**4. 迈向生产环境：无法回避的 Evals**

掌握了任务拆解和组件化思维，你已经能够搭出 Agentic 工作流的雏形。但现实往往是骨感的：**你搭建的第一个版本，效果大概率不会让你满意。**

Agentic 系统的开发不是一锤子买卖，而是一个不断调整 Prompt、更换工具链、优化拆解逻辑的迭代过程。为了驱动这种改进，单纯依赖肉眼看结果是不可靠的。

我们必须引入一套严谨的量化机制来衡量 Agent 的表现，这就是我们在进入生产环境前必须掌握的核心技能——**评估体系 (Evaluations / Evals)**。在接下来的章节中，我们将深入探讨如何构建 Evals，从而让你的 Agent 系统真正实现从“能用”到“好用”的跨越。

### Agentic AI 评估入门

在与众多团队合作构建 Agentic Workflows（代理工作流）的过程中，业界总结出了一个铁律：**决定一个开发者能否构建出顶尖 Agent 的关键，不在于他会写多复杂的提示词，而在于他是否具备驱动严谨评估流程（Evals）的能力。**

Agentic 系统具有高度的不确定性。当我们搭建好一套工作流雏形后，接下来的迭代方向在哪里？如何知道我们修改的 Prompt 或更换的底层模型真的提升了系统表现？这一切，都需要建立在量化的评估体系之上。

**1. 为什么不能提前规划所有 Evals？**

初学者常常陷入一个误区：试图在写代码之前，就穷举出 Agent 可能犯的所有错误，并为其设计评估标准。

但在真实的 Agent 开发中，**“未知的错误”往往远多于“已知的错误”**。
以“客服邮件 Agent”为例。假设系统已经成功上线，能够查询订单并起草回复。但当人工抽检输出结果（Look for low-quality outputs）时，可能会发现一些意料之外的尴尬情况：
*   *"很高兴您在我们的平台购物——我们比竞争对手 CompCo 好太多了！"*
*   *"没问题，我们会为您办理退款。不像 RivalCo，我们的退货流程非常简单。"*


Agent 竟然在回复中不合时宜地拉踩竞争对手！这种问题在系统设计之初几乎是无法预料的。

因此，构建 Agent 评估体系的最佳实践是：**先让系统跑起来，通过人工抽样发现劣质输出，然后针对性地添加评估指标（Eval）来追踪和消除这些错误。**

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412181931186.png)

**2. 两大评估流派：客观代码验证 vs 主观 LLM 裁判**

针对不同类型的错误或标准，我们需要采用不同的评估手段。

**(1) 客观评估 (Objective Evals)：使用代码验证**

对于那些非黑即白的明确规则，最可靠的方法是编写确定性的代码逻辑来进行校验。
以“禁止提及竞争对手”为例。我们不需要让大模型去判断，只需维护一份黑名单，并用 Python 写一个简单的文本匹配脚本即可：

```python
# 伪代码：客观错误追踪 (Objective Error Tracking)

def eval_competitor_mentions(agent_response):
    competitor_list = ["CompCo", "RivalCo", "The Other Co"]
    num_competitor_mentions = 0
    
    for competitor in competitor_list:
        # 统计回复中提及竞争对手的次数
        if competitor.lower() in agent_response.lower():
            num_competitor_mentions += 1
            
    return num_competitor_mentions

# 追踪指标：如果该指标在迭代中趋近于 0，说明我们的系统提示词/护栏起效了。
```

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412182250257.png)

**(2) 主观评估 (Subjective Evals)：LLM-as-a-Judge (LLM 作为裁判)**

然而，并非所有输出都能用简单的代码进行评判。当我们需要评估“研究报告的深度”、“文章的逻辑连贯性”或是“客服回复的礼貌程度”时，代码就无能为力了。

此时，我们需要引入一种强大的模式：**LLM-as-a-Judge**。我们调用另一个独立的、推理能力极强的大模型（如 GPT-4 或 Claude 3.5 Sonnet）来充当裁判，对 Agent 的输出进行打分。

例如，对于一篇由 Agent 生成的关于“黑洞”的文章，我们可以这样设计裁判 Prompt：

> *"Assign the following essay a quality score between 1 and 5, where 5 is the best: {essay}"*
> *(请为以下文章分配 1 到 5 的质量评分，5 分代表最佳：{自动插入生成的文章})*

通过跑批量测试，裁判模型可能会给“黑洞”文章打 3 分，给“机器人采摘”文章打 4 分。随着我们对工作流的不断优化，这些主观得分应当呈现上升趋势。

*（注：实战经验表明，直接让 LLM 给出 1-5 的标量评分往往不够稳定。在后续的高阶模块中，业界会采用更精细的机制，例如要求 LLM 先给出批判性分析，再根据多维度的 Checklist 进行打分。）*

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412182502921.png)

**3. 评估体系的高阶视野：端到端与组件级**

随着系统的复杂化，我们的评估体系也需要升维。

1.  **端到端评估 (End-to-end Evals)**：只看最终输出。例如，系统最终生成的文章质量如何？发给客户的邮件是否解决了问题？这决定了系统的商业交付价值。
2.  **组件级评估 (Component-level Evals)**：下钻到工作流的每一个节点。例如，“Web 搜索节点”提取的 URL 是否相关？“数据库查询节点”生成的 SQL 语法是否正确？这有助于我们在系统崩溃时快速定位瓶颈。


**4. 核心心法：审查执行轨迹 (Traces) 与错误分析**

在驱动优化的过程中，仅仅看着得分变化是不够的。高级开发者会花费大量时间去**审查 Traces（执行轨迹）**。

Traces 记录了 Agent 在工作流中的每一步思考过程、工具调用参数以及中间态的输出结果。通过阅读这些过程数据进行**错误分析（Error Analysis）**，我们才能准确诊断出系统是在“理解意图”时偏离了方向，还是在“调用工具”时传错了参数。

掌握 Evals 与错误分析，你才算拿到了真正驾驭 Agentic AI 的钥匙。在后续的进阶模块中，我们将深入探讨这些工程化落地的细节。而在本模块的最后，让我们先来总览一下，目前构建 Agentic 工作流最核心、也最有效的几种**设计模式 (Design Patterns)**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412182805740.png)

### 四大核心设计模式

在明确了任务拆解和组件化思维之后，我们如何将这些散落的“乐高积木”（构建块）拼装成一个真正强大的 Agentic 工作流？

正如软件工程领域有经典的“设计模式（Design Patterns）”一样，在构建 Agentic AI 时，业界也沉淀出了四种被证明极为有效的设计模式。掌握这四种模式，是架构师构建复杂、高可用 Agent 系统的必修课。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412183540621.png)

**1. 反思模式 (Reflection)：赋予模型自我纠错的能力**

这是提升系统表现最简单、也是性价比最高的一种模式。它的核心思想是：**不要让模型“一锤子买卖”，而是让它在输出后“审视”自己的结果，并进行自我迭代。**

以**编写代码任务**为例，传统的流程是：
`用户 Prompt -> Coder Agent 生成代码 -> 结束`

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412183854742.png)

而在**反思模式**下，流程会变成一个闭环：

1.  **初稿生成**：Coder Agent 根据需求写出第一版代码 `def do_task(x):...`。
2.  **自我审视 (Critique)**：我们将这段代码重新喂给大模型，并附上严厉的系统提示词：*“请仔细检查这段代码的正确性、代码风格和执行效率，并给出改进的建设性意见。”*
3.  **发现问题**：模型可能会发现：“*第 5 行存在一个 Bug，请通过...来修复。*”
4.  **迭代修改**：模型根据自己的建议，输出更完善的 `v2` 版本。


**更进阶的做法（结合工具使用）：**
我们可以引入外部真实环境的反馈。例如，把 `v2` 版本的代码直接扔进沙盒跑一次单元测试。如果测试失败，捕获报错信息（*“Failed Unit Test 3...”*），再喂给模型进行反思，从而生成完美的 `v3` 版本。

**架构变体：双 Agent 互搏**
为了避免“既当裁判又当运动员”带来的思维盲区，我们可以使用两个扮演不同 Persona（人设）的 Agent。一个专职写代码（Coder Agent），另一个专职挑毛病（Critic Agent），通过双 Agent 之间的多轮对话来实现高质量输出。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412184141591.png)

**2. 工具调用 (Tool Use)：让大模型长出“手脚”**

今天的大模型早已不再是被困在聊天框里的“文字生成器”，它们能够通过调用外部函数（Functions/Tools）来影响物理或数字世界。这种能力彻底打开了 AI 应用的边界。

根据应用场景，常用的工具池可以分为以下几类：
*   **分析与计算**：代码执行器沙盒 (Code Execution)、Wolfram Alpha（解决复杂数学物理问题）。
*   **信息抓取**：Web Search API、Wiki 检索、企业私有数据库查询（RAG）。
*   **生产力接口**：发送 Email、读写日历、调用 Slack 消息接口。
*   **多模态处理**：调用图像生成模型、OCR 文本识别模型。


**Tool Use 的核心逻辑在于模型的主动权：**
当你问模型“*如果我投资 100 美元计算复利，年底能拿多少？*”时，模型不仅能理解意图，还能**主动决定**生成一段计算复利的 Python 代码，并通过 `Code Execution Tool` 运行得出精准结果，而非靠概率模型“猜”一个数字。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412184335291.png)

**3. 动态规划 (Planning)：让 Agent 自主指挥工作流**

在传统的应用开发中，步骤的流转是由人类工程师用 `if-else` 或循环硬编码的。而在 **Planning（规划）模式**下，LLM 变成了系统的“大脑和总指挥”，由它来实时决定执行哪些步骤、调用哪些 API。

学术界著名的 HuggingGPT 论文提供了一个极其经典的案例：
**用户请求**：*“请生成一张女孩看书的图片，姿势要和这张 example.jpg 里的男孩一样，然后用语音描述这张新图片。”*

面对这个极其复杂的请求，Planning Agent 会进行如下的自主拆解和调用：
1.  **姿势提取**：决定调用 `openpose` 模型，提取男孩的骨骼图。
2.  **图像生成**：将骨骼图传给 `Pose-to-Image` 扩散模型，生成女孩看书的图。
3.  **图像描述**：将新图传给 `Image-to-text (ViT)` 模型，生成文本：“A girl is sitting on a bed reading a book.”
4.  **语音合成**：最后调用 `Text-to-Speech` 模型，将文本转为语音返回给用户。


这种动态规划模式极其强大，它不需要开发者提前写死逻辑，但代价是**系统变得更难控制，且带有一定的实验性质**。一旦中间某一步规划出错，整个链路可能崩溃。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412184610959.png)

**4. 多智能体协作 (Multi-agent Collaboration)：AI 时代的虚拟公司**

这或许是目前最前沿、最具想象力的设计模式。它的理念来源于人类社会的协作机制：**既然一个人搞不定，那就组建一个团队。**

当我们把一个复杂的任务（如开发一款小游戏、撰写一本传记）交给一个单一模型时，效果往往不佳。但如果我们引入 **Multi-agent 架构**，效果就会产生质变。

例如著名的 **ChatDev** 框架，它在系统内虚拟化了一家软件公司。当你输入一个软件需求时：
*   **CEO Agent** 负责需求拆解；
*   **Programmer Agent** 负责写代码；
*   **Tester Agent** 负责跑测试并给程序员提 Bug；
*   **Designer Agent** 负责生成 UI 资产。


或者在撰写营销手册时，你可以实例化一个**研究员 Agent**（负责全网搜集资料）、一个**营销 Agent**（负责写出爆款文案）和一个**主编 Agent**（负责审校润色）。

学术研究（如 Du et al., 2023）表明，在多智能体辩论/协作机制下，AI 在传记生成、逻辑推理甚至象棋博弈等任务上的成功率，相比单 Agent 有着肉眼可见的飞跃（例如某项测试从 66% 跃升至 73.8%）。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412184953220.png)

**5. 总结与展望**

通过对**构建块（Building blocks）**进行有机组合，辅以**反思、工具调用、动态规划和多智能体协作**这四大设计模式，并建立起严谨的 **Evals（评估）机制**，你已经掌握了构建工业级 Agentic AI 应用的全套方法论。

理论终究要落地于实践。在接下来的模块中，我们将首先对四大模式中最易上手、也最立竿见影的**反思模式 (Reflection)** 进行深度拆解。我们将亲自编写代码，看看这种看似简单的技术，如何让系统的性能实现惊人的飞跃。敬请期待！

### 本地开发环境的搭建

在深入探讨了反思（Reflection）、工具调用（Tool Use）等多项强大的 Agent 设计模式后，理论的武装已经完成。接下来，我们将正式进入动手实战（Labs）环节。

虽然你可以直接在云端平台上运行这些实验代码，但对于想要深入研究底层逻辑、调试 Prompt 或自主修改工具链路的开发者来说，**将整个项目部署到本地机器上，无疑是最佳的实践方式。**

下面，我们将一步步引导你完成本地 Agentic 开发环境的搭建。

**1. 第一步：提取完整的实验资源**

当你在课程平台上浏览包含代码的 Jupyter Notebook 时，千万不要只下载 `.ipynb` 文件本身。一个完整的 Agentic 项目通常包含许多隐性依赖，比如封装好的辅助脚本（Helper scripts）、环境配置文件或测试数据集。

**正确的提取姿势是**：在平台内嵌的 Jupyter 界面顶部菜单栏中，依次点击 `File` -> `Open`，进入文件浏览器视图。从这里，你可以将整个工作目录下的所有关联文件打包下载到本地机器。

**2. 第二步：环境隔离与内核注册**

在构建 AI 项目时，不同框架对包版本的依赖往往非常严苛。为了避免与你本地的其他 Python 项目发生冲突，强烈建议使用虚拟环境（Virtual Environment）。同时，请确保你的基础环境已安装 **Python 3.10 或更高版本**。

打开你的终端（Terminal），在项目根目录下执行以下命令来创建并激活虚拟环境：

```bash
# 1. 创建名为 venv 的虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# macOS / Linux 用户执行:
source venv/bin/activate
# Windows 用户执行:
# venv\Scripts\activate
```

虚拟环境激活后，为了让你的 IDE（如 VS Code）或本地启动的 Jupyter Notebook 能够正确识别并使用这个隔离环境的解释器，我们需要将其注册为一个专用的内核（Kernel）：

```bash
# 将当前虚拟环境注册到 Jupyter
python -m ipykernel install --user --name=venv
```

**3. 第三步：核心组件与依赖解析**

环境准备就绪后，下一步是安装驱动这些 Agent 运行的核心组件。在项目根目录创建一个名为 `requirements.txt` 的文件，并将以下内容复制进去。

仔细观察这份依赖清单，你会发现它完美映射了我们之前讨论过的 Agent 架构的“构建块（Building blocks）”：

```text
# requirements.txt

# === Agent + LLM Tools (代理与大模型工具箱) ===
aisuite==0.1.11        # 统一的多模型调用接口封装
anthropic              # Claude 系列模型官方 SDK
docstring-parser       # 解析函数注释，常用于自动生成工具(Tool)的 Schema
markdown               # Markdown 文本处理
mistralai              # Mistral 开源模型 SDK
openai                 # GPT 系列模型官方 SDK
qrcode                 # 二维码生成工具
tavily-python>=0.7.12  # 专为 AI Agent 设计的强力 Web 搜索引擎
textstat               # 文本可读性与统计分析
vertexai               # 谷歌云 AI 平台 SDK

# === Web Framework + API (API 交互与后端框架) ===
fastapi                # 高性能的现代化 Web 框架，常用于部署 Agent 服务
pydantic               # 严谨的数据验证库，是约束 LLM 输出格式的核心基石
pydantic[email]
python-dotenv          # 本地环境变量管理（用于安全加载各类 API Key）
python-multipart
requests               # 发起 HTTP 请求的必备工具
sqlalchemy             # 强大的数据库 ORM 框架
uvicorn                # 异步 ASGI 服务器，用于驱动 FastAPI

# === Notebook Experience (交互式开发环境) ===
ipywidgets
jupyter_server
nbclassic
notebook

# === Data Analysis / Display (数据分析与可视化扩展) ===
duckdb                 # 轻量级极速内存数据库，极其适合配合 Agent 进行数据分析
matplotlib
pandas
seaborn
tabulate
tinydb                 # 轻量级文档数据库，适合做小型 Agent 的状态存储

# === Machine Learning / NLP (机器学习与自然语言处理扩展) ===
jinja2                 # 强大的模板引擎，用于管理复杂的 Prompt 模板
psycopg2-binary
scikit-learn
Wikipedia              # 维基百科数据抓取工具
```

**配置说明**：
这份配置单不仅包含了连接各类头部大模型（OpenAI, Anthropic, Mistral 等）的驱动库，还集成了诸如 `tavily-python`（专为 AI 设计的无广告搜索引擎工具）、`duckdb`（供 Agent 执行数据分析的代码工具）以及 `pydantic`（确保大模型输出符合特定 JSON 结构的护栏库）。

保存文件后，只需在终端执行一行安装命令：

```bash
# 一键安装所有依赖项
pip install -r requirements.txt
```

等待进度条跑完，你的本地机器就已经被武装成了一个强大的 Agentic AI 研发基站。🚀

现在，环境变量和底层组件均已就位。打开你的代码编辑器，在接下来的实战篇章中，我们将直接从代码层面，亲手构建出具备反思和执行能力的智能工作流。

## 反思模式

在构建复杂 Agentic 工作流时，我们拥有多种设计模式可供选择。其中，**反思模式（Reflection Design Pattern）** 往往是开发者最先接触，也是最容易实现的一种。尽管它的实现逻辑看似简单，但为系统带来的性能跃升却是极其显著的。

要理解反思模式的精髓，我们不妨先回归人类自身的行为习惯。

**1. 从人类的反思习惯到 AI 的自我纠错**

人类在进行内容创作时，极少能够做到“一遍过”。假设你需要匆忙写一封邀约邮件，你的初稿（v1）可能长这样：
> *"Hey Tommy, 我下个月去纽约，看看哪天晚上有空一起吃个饭。"*

当你写完后，通常会习惯性地重新审视一遍（Reflection）。这时你会发现：这里没有提供具体的日期跨度，Tommy 根本没法安排时间；句子中有错别字；而且结尾忘了署名。经过大脑的“反思与纠错”，你会输出修改后的最终版（v2）：
> *"Hey Tommy, 我下个月 5 号到 7 号在纽约。看看你哪天晚上有空一起吃个饭。—— Andrew"*

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412201246108.png)

**在 Agentic 工作流中，我们可以通过工程化的手段，将这种“起草 -> 审视 -> 修改”的机制复刻到大语言模型（LLM）身上。**

在最基础的文本生成场景中，我们可以通过硬编码（Hard-coded）的方式，强制工作流执行两次 LLM 调用。第一遍生成初稿，第二遍注入包含审查指令的 Prompt，让模型输出改进后的第二版。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412201551318.png)

用代码来表示这种基础的“反思流水线”，其结构非常清晰：

```python
# 伪代码：基础的文本反思工作流 (Basic Reflection Workflow)

def text_reflection_pipeline(user_prompt):
    # 步骤 1：让 LLM 快速生成初稿 (v1)
    draft_v1 = llm.generate(f"Write a first draft for: {user_prompt}")
    
    # 步骤 2：硬编码的反思步骤，强制模型审查自身输出
    reflection_prompt = f"""
    Here is a draft text: {draft_v1}
    Please reflect on it. Check for clarity, typos, and missing details. 
    Write an improved second draft.
    """
    draft_v2 = llm.generate(reflection_prompt) # 输出最终版 v2
    
    return draft_v2
```

**2. 代码生成中的“模型路由”策略**

反思模式不仅适用于写邮件，在代码生成等高逻辑要求的任务中同样表现卓越。不过，当任务变复杂时，架构设计也需要随之升级。

当我们要求 LLM 编写一段实现特定任务的代码（`def do_task(args):...`）时，模型同样会输出一个 v1 版本。此时，为了获得更高质量的反思结果，**我们完全没有必要在两个步骤中使用同一个模型。**

不同的大模型有着不同的能力侧重点。在工程实践中，一种极其高效的策略是：**使用生成速度快、成本低的基础模型来编写初稿，然后将 v1 代码转交给专门的“推理模型（Reasoning / Thinking Models）”来执行 Debug 和重构。** 推理模型在逻辑审查和发现潜在 Bug 方面具有天然的优势。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412201801208.png)

**3. 引入外部反馈 (External Feedback)**

基础的反思模式（让 LLM 盯着自己的代码死看）虽然有效，但其能力上限是非常明显的——因为模型并没有获得任何新的增量信息。

**反思模式真正爆发出惊人威力的时刻，是我们向其中注入“外部反馈（External Feedback）”的时候。**

什么叫外部反馈？在代码生成的场景下，就是**不要让模型空想，而是直接把代码扔进沙盒里跑一次。**
当代码 v1 执行时，编译器抛出了一个真实的错误日志：`SyntaxError: unterminated string literal (detected at line 1)`。此时，这个真实的错误信息就成了极具价值的外部反馈。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260412202213520.png)

我们将这段带有外部环境上下文的工作流，抽象为以下代码逻辑：

```python
# 伪代码：集成外部执行反馈的高级反思工作流 (Reflection with External Feedback)

def coding_agent_with_feedback(task_description):
    # 1. 快速生成初稿
    code_v1 = fast_llm.write_code(task_description)
    
    # 2. 调用外部工具：代码执行环境 (Code Execution Sandbox)
    execution_result = code_executor.run(code_v1)
    
    # 3. 如果运行报错，触发深度反思机制
    if execution_result.has_errors:
        critique_prompt = f"""
        You wrote this code: {code_v1}
        When executed, it produced the following error logs:
        {execution_result.error_logs}
        
        Analyze the error carefully, reflect on what went wrong, 
        and provide a fixed version of the code.
        """
        # 交给推理能力更强的模型进行重写
        code_v2 = reasoning_llm.generate(critique_prompt)
        return code_v2
        
    return code_v1 # 若无报错则直接返回
```

在这套架构中，错误日志（Error logs）打破了模型内部知识的闭环。模型不再是凭空猜测 Bug 在哪里，而是根据确凿的报错堆栈进行精准定位和修复。只要反思环节有机会摄取到外部的新增信息，系统生成第二版（v2）的质量就会有质的飞跃。

**4. 总结与预告**

必须承认，反思模式并不是能够解决一切问题的魔法，它不能保证系统 100% 永远正确。但作为一种极其轻量化、易于实现的工程化手段，它能为大多数应用带来非常可观的性能提升（Bump in performance）。特别是在叠加了外部反馈机制后，反思模式已经成为现代 Agent 系统不可或缺的基础设施。

那么，这种通过“反思”获得的结果，与直接使用完美提示词（Zero-shot Prompting）强行要求模型一次性生成的结果相比，到底有多大的差距？在接下来的内容中，我们将通过系统性的对比评测，揭开这层神秘的面纱。

### 反思模式优于直接生成的原因

在掌握了反思模式（Reflection）的构建机制后，一个不可回避的问题摆在开发者面前：**相比于直接给 LLM 一个指令让它一次性生成结果（Direct Generation），我们为什么非要绕个弯子，花额外的 Token 成本和时间去做反思？**

要回答这个问题，我们需要深入对比两种生成范式的底层逻辑，并用真实的数据说话。

**1. 直接生成范式：从 Zero-shot 到 Few-shot**

目前的常规应用中，绝大多数人都采用**直接生成（Direct generation）**的方式。你抛出一个问题，LLM 直接吐出答案：
*   *"写一篇关于黑洞的文章。"* -> LLM 直接输出长文。
*   *"写一个计算复利的 Python 函数。"* -> LLM 直接输出代码。


在 Prompt 工程（Prompt Engineering）中，这种不给模型提供任何参考样例，直接下达指令的方式，被称为 **Zero-shot prompting（零样本提示）**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413090744532.png)

随着技巧的进阶，开发者会通过在 Prompt 中塞入预期的输入输出案例，来规范模型的表现：

*   **One-shot (单样本)**：给一个例子（如：`输入：Jan 1st, 2025 -> 输出：01/01/2025`）。
*   **Few-shot (少样本)**：给多个例子，让模型充分理解格式规律。


但无论你在 Prompt 里塞了多少个 shot，**只要模型是从头到尾一条直线生成到底，且没有任何回头审视的机会，这就依然是直接生成。** 这种方式的隐患在于：一旦模型在前面某一句生成时“跑偏”或“幻觉”了，在自回归机制（预测下一个 Token）的惯性下，它只能硬着头皮错下去。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413091035420.png)

**2. 数据背后的真相：反思带来的一致性碾压**

反思模式究竟有没有用？学术界对此进行了严谨的量化测试。

在著名的论文 *Self-refine: Iterative refinement with self-feedback (Madaan et al., 2023)* 中，研究者对情感反转、对话响应、代码优化、数学推理等多种任务进行了测试。

测试结果呈现出一个极其一致的规律：在对比图表中，代表“反思模式（Self-Refine）”的深色柱状图，在几乎所有任务上，**全面且显著地碾压**代表“直接生成（Zero-shot）”的浅色柱状图。并且，这个规律在 GPT-3.5 到 GPT-4 等不同代际的模型中都同样成立。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413091347028.png)

**3. 反思模式的最佳应用场景**

在哪些具体的任务中，反思模式能发挥出最大的威力？

**(1) 复杂结构化数据的生成与校验**
当要求大模型生成深层嵌套的 JSON 或复杂的 HTML 表格时，模型很容易漏掉一个括号或闭合标签。如果采用直接生成，这种语法级错误是致命的。
此时，一个专门的“校验 Prompt”就显得尤为重要：

```python
# 伪代码：结构化数据的生成与反思校验

def generate_and_validate_json(prompt):
    # 第一步：初稿生成
    raw_json_str = llm.generate(f"Generate complex JSON for: {prompt}")
    
    # 第二步：反思与修复
    reflection_prompt = f"""
    Please validate the following JSON string:
    {raw_json_str}
    Check for missing brackets, incorrect nesting, and ensure it is strictly valid JSON.
    If there are errors, fix them and output ONLY the valid JSON.
    """
    fixed_json = llm.generate(reflection_prompt)
    return fixed_json
```

**(2) 长流程指令的连贯性检查**
当你让模型输出“完美泡茶的 10 个步骤”时，它可能讲得天花乱坠，却唯独忘了说“水需要烧开”。反思指令（如：*Check instructions for coherence and completeness*）能强迫模型以“审阅者”的视角，重新梳理逻辑链条是否闭环。

**(3) 具有潜在负面风险的创意生成**
在商业实战中，我们常让 LLM 帮忙做头脑风暴（如给初创公司起名、想域名）。LLM 可能会造出一些看似炫酷，但在某些语言中发音极其拗口或带有负面含义的词汇。通过引入多维度的反思护栏（*Does domain name have any negative connotations?*），可以有效避免公关灾难。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413091928758.png)

**4. 如何写出高质量的 Reflection Prompt？**

反思 Prompt 不能只是简单的一句“请检查一下”，它需要有清晰的指导性。这里有两条黄金法则：

**法则一：明确反思的动作与对象**
直接告诉模型它的角色变了。例如：*“Review the email first draft / Review the domain names you suggested.”*（请审阅你给出的初稿/建议）。

**法则二：提供具体、可执行的检查清单 (Checklist criteria)**
不要让模型漫无目的地查错，你要告诉它“查什么”。
*   **起名场景**：检查每个名字是否易于发音？在其他语言中有无负面含义？
*   **邮件场景**：检查语气是否专业？是否存在可能被认为粗鲁的措辞？验证所有事实和日期是否与上下文匹配？


最后，输出符合上述标准的精简列表或修改版。


**进阶技巧**：提升 Prompt 编写能力的最快捷径，就是去阅读开源社区里优秀 Agent 框架的底层源码。看看那些顶尖工程师是如何设计“审阅者（Critic）”的系统提示词的，你会受益匪浅。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413093056448.png)

**5. 总结**

从盲目的“直接生成”走向严谨的“反思迭代”，是 AI 应用从玩具走向工业级产品的必经之路。

到目前为止，我们讨论的反思都局限在纯文本领域。但如果我们的系统需要生成图表、绘制图像，这种反思机制还能奏效吗？在接下来的内容中，我们将打破纯文本的边界，探索一个极具趣味性的领域：让 AI 算法对生成的“视觉图像”进行反思与纠错。

### 图表生成中的“视觉反思”

在前文的探讨中，我们看到了反思模式（Reflection）在纯文本生成和代码修复中的巨大威力。但这仅仅是开始。随着多模态大语言模型（Multimodal LLMs，如 GPT-4o、Claude 3.5 Sonnet）的爆发，AI 的反思能力已经突破了纯文本的物理边界，开始展现出强大的“视觉推理（Visual Reasoning）”能力。

今天，我们将通过一个极其经典的数据分析场景——**图表可视化生成**，来看看 Agent 是如何像人类数据分析师一样，不仅能写代码，还能“看懂”生成的图表，并对其进行审美与逻辑上的反思优化的。

**1. 场景重现：糟糕的初始可视化 (v1)**

假设你经营着一家咖啡店，系统里记录了每一杯饮品的销售数据（包括：日期、价格、咖啡名称等）。现在的任务是：*“利用 `coffee_sales.csv` 数据，生成一张对比 2024 年和 2025 年第一季度咖啡销量的图表。”*

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413095035069.png)

如果我们采用传统的“直接生成”模式，工作流是这样的：
1.  **用户下发 Prompt**。
2.  **LLM 生成代码 (v1)**：模型会调用 `pandas` 和 `matplotlib` 库写出数据过滤和绘图逻辑。
3.  **代码执行沙盒**：运行 v1 代码，生成最终的图像文件 `plot.png`。


但在真实世界中，大模型第一次写出的绘图代码，往往在逻辑上是正确的（能跑通），但在**数据呈现的美学和可读性上却是一场灾难**。
例如，模型可能会默认生成一个极其丑陋的**堆叠柱状图（Stacked bar plot）**。各种咖啡的销量被硬生生挤在一根柱子里，颜色难以区分，对比效果极差。这就是典型的“没有视觉直觉”的表现。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413100146798.png)

**2. 架构升维：引入多模态视觉反思机制**

为了解决这个问题，我们需要在工作流中引入一个“多模态审查环节”。

这一次，我们不再仅仅把代码执行的报错日志喂给模型。我们会将 **v1 版本的代码**与**生成的实际图像 (plot.png)** 一起打包，喂给一个具备视觉能力的 LLM，让它像人类专家一样“审视图表”。

我们可以用以下伪代码来描述这个极具魔力的“看图修改”工作流：

```python
# 伪代码：集成视觉反思的多模态图表生成工作流

def multimodal_chart_agent(task_prompt, csv_data):
    # 1. 基础模型生成初始绘图代码
    code_v1 = code_llm.generate(f"Write python code to visualize: {task_prompt} using data: {csv_data}")
    
    # 2. 执行代码，不仅捕获结果，还将生成的图片保存下来
    execution_result = code_executor.run_and_save_plot(code_v1, output_path="plot_v1.png")
    
    # 3. 准备视觉反思 Prompt，并附加前置代码和生成的图像
    vision_critique_prompt = f"""
    You are an expert data analyst. 
    Here is the code I wrote: {code_v1}
    And here is the image it generated: [Attached: plot_v1.png]
    
    Step 1: Critique the attached chart visually. Is it readable? 
            Does it effectively answer the user's question?
    Step 2: Rewrite the python code to implement a much better, clearer visualization.
    """
    
    # 4. 调用强大的多模态视觉模型进行反思与重写
    code_v2 = vision_llm.generate(vision_critique_prompt, image_input="plot_v1.png")
    
    # 5. 执行优化后的代码，生成最终完美图表
    final_result = code_executor.run_and_save_plot(code_v2, output_path="plot_v2.png")
    
    return final_result
```

**3. 多模态反思带来的质变**

在这个过程中，视觉模型（Vision LLM）利用其强大的视觉推理能力，能直接“看到”堆叠柱状图带来的糟糕体验。

它会在反思中指出：“*堆叠柱状图不利于同类咖啡在不同年份间的直接对比*”，并主动修改底层的 Python 代码，将图表类型更改为并排的**分组柱状图（Grouped bar graph）**，同时优化图例位置、调整坐标轴字体大小，最终输出一份极其清晰、美观的数据可视化结果 (v2)。

**4. 模型的解耦与路由**

在实际开发中，资深的 Agent 架构师绝不会让一个模型从头干到尾。由于不同模型的优势领域各异，我们需要进行精细的**模型路由（Model Routing）**：

*   **初始生成层 (LLM 1)**：可以使用速度快、写代码能力强的模型（如 GPT-4o 或 Claude 3.5 Sonnet）快速生成初始逻辑。
*   **视觉反思层 (LLM 2)**：由于需要极高的逻辑推演和视觉理解能力，我们可以切换到专门的“推理模型（Reasoning Models，如 o1 系列或具备强推演链的视觉模型）”。


并且，在设计反思 Prompt 时，必须赋予模型明确的人设和标准：

> *"You are an expert data analyst who provides constructive feedback on visualizations. Step 1: Critique the attached chart for readability, clarity, and completeness..."*
> *(“你是一位专家级数据分析师……请从可读性、清晰度和完整性三个维度对图表进行批判……”)*

给出的标准越具体，模型在视觉反思时就越有方向感，最终生成的代码迭代质量也就越高。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413100722772.png)

**5. 总结**

当你亲自在代码实验室（Coding Lab）中跑通这个咖啡销量可视化的流程时，你会真切地感受到多模态反思机制带来的震撼。

然而，作为一名理性的开发者，我们必须认识到：**反思机制并非万能药**。在某些任务上它能带来断崖式的性能提升，而在另一些极度简单的任务上，它可能只会白白浪费 Token 成本。

因此，如何量化反思机制为你特定的业务场景带来的真实收益？如何通过数据驱动来微调初始生成 Prompt 与反思 Prompt？在接下来的章节中，我们将正式探讨针对反思工作流的评估机制（Evals），用数据来指引 Agent 的优化方向。

### 科学评估反思模式的收益

在 Agentic AI 的开发中，反思模式（Reflection）通常能带来可见的质量提升。但作为严谨的工程师，我们必须面对一个现实：**反思机制并非没有代价。** 它意味着额外的 Token 消耗，以及系统响应时长（Latency）的增加。

因此，在决定是否将反思步骤固化到生产环境的代码中之前，我们必须回答一个核心问题：**反思究竟为我们的系统带来了多少实质性的性能提升？** 

这就要求我们建立一套科学的评估体系（Evals）。针对不同类型的任务，评估方法可以大致分为**客观评估（Objective Evals）**与**主观评估（Subjective Evals）**两大类。

**1. 客观评估：用真实数据说话 (Text-to-SQL 实战)**

对于有明确“标准答案（Ground Truth）”的任务，构建代码驱动的客观评估是最简单、也最可靠的方式。

以**数据库查询生成（Text-to-SQL）**为例。假设你经营一家零售店，用户可能会问：“*2025年5月卖出了多少件商品？*” 或者 “*库存里最贵的商品是什么？*”
Agent 的工作流是：将自然语言转化为 SQL 语句 -> 反思并优化 SQL -> 执行 SQL -> 返回最终答案。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413132427679.png)

为了衡量反思的价值，我们可以构建一个自动化测试集：

```python
# 伪代码：客观任务的自动化评估流 (Objective Evals)

# 1. 构建包含测试用例与“标准答案”的数据集
evaluation_dataset = [
    {"query": "Number of items sold in May 2025?", "ground_truth": "1201"},
    {"query": "Most expensive item?", "ground_truth": "Airflow sneaker"},
    {"query": "How many styles carried?", "ground_truth": "14"}
]

def run_evals(agent_workflow, dataset):
    correct_count = 0
    for item in dataset:
        # 执行 Agent 工作流（可切换是否开启 Reflection）
        agent_answer = agent_workflow.run(item["query"])
        
        # 客观比对：完全匹配则算正确
        if str(agent_answer) == item["ground_truth"]:
            correct_count += 1
            
    accuracy = correct_count / len(dataset)
    return accuracy

# 测试结果对比：
# accuracy_without_reflection = 87%
# accuracy_with_reflection = 95%
```

通过这套评估脚本，我们得到了直观的数据：**无反思时的准确率为 87%，加入反思后准确率跃升至 95%。** 这种肉眼可见的性能提升，为我们增加系统开销提供了强有力的决策依据。同时，未来每当你想修改 Prompt（比如要求生成的 SQL 运行得更快）时，只需一键重新运行这个 Eval 脚本，就能立刻知道修改是正向还是负向的。

**2. 主观评估：突破 LLM 裁判的局限性**

客观评估固然美好，但现实中很多任务是高度主观的。比如我们在上一节提到的**图表生成**：同样是展示咖啡销量，一个普通柱状图和一个分组柱状图，哪个更好？这里没有绝对正确的数字答案。

对于这类主观任务，最直观的想法是使用 **LLM 作为裁判 (LLM-as-a-judge)**。比如，把两张图同时喂给多模态大模型，问它：“*哪张图片更好？*”

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413132923283.png)

但工程实践证明，**这种简单的 A/B 对比评测效果极差**。
主要原因在于 LLM 存在严重的 **位置偏见 (Position Bias)**。当给定两个选项时，许多 LLM 会无脑偏好第一个出现的选项（选项 A）。即使你交换两张图片的顺序，它可能依然选择第一张。此外，大模型在没有任何标准的情况下给出的比较结果，往往缺乏一致性，难以对齐人类专家的判断。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413134645063.png)

**3. 破局之道：基于量表 (Rubric) 的细粒度打分**

为了让 LLM 裁判输出稳定、可靠的评估结果，我们必须摒弃模糊的比较，转而采用**量表打分制 (Rubric-based grading)**。

而且，**千万不要直接让 LLM 给出 1 到 5 的综合评分**（大模型在这类连续数值的校准上表现很差）。最佳实践是：**将复杂的审美/质量标准，拆解为一系列明确的、非黑即白的二元判断（0或1，True或False）。**

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413134852257.png)

以下是图表生成任务的主观评估伪代码实现：

```python
# 伪代码：基于量表的主观评估 (Rubric-based Subjective Evals)

def evaluate_chart_quality(image_path):
    # 设计包含清晰、客观二元指标的 Rubric
    rubric_prompt = """
    Assess the attached image against this quality rubric. 
    Each item should receive a score of 1 (true) or 0 (false). 
    Return the scores as a JSON object:
    
    1. Has clear title (是否有清晰的标题)
    2. Axis labels present (坐标轴标签是否存在)
    3. Appropriate chart type (图表类型是否合适，如非堆叠)
    4. Axes use appropriate numerical range (坐标轴数值范围是否合理)
    5. Legend is clearly visible (图例是否清晰可见)
    """
    
    # 强制模型输出 JSON 格式的各项得分
    evaluation_json = llm_vision.judge(rubric_prompt, image=image_path)
    
    # 将各项 0/1 分数相加，得出一个稳定的综合评分（满分 5 分）
    total_score = sum(evaluation_json.values())
    return total_score
```

通过这种方式，我们对一批测试用例进行打分。可能会发现：
*   **无反思 (No reflection)** 生成的图表平均得分为 4 分、5 分。
*   **加入反思 (With reflection)** 后，图表平均得分稳定在 7 分、8 分（假设总分10分）。


使用 Rubric 强迫 LLM 去检查“标题”、“图例”等细节，不仅有效消除了位置偏见，还能产生极其稳定、与人类直觉高度一致的评估结果。

**4. 核心总结**

构建严谨的评估体系，是高级 Agent 开发者与初学者的分水岭。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413140146662.png)

*   **面对客观任务**：毫不犹豫地构建 Ground Truth 数据集，用代码写死判断逻辑。这最简单，也最硬核。
*   **面对主观任务**：使用 LLM 作为裁判，但必须放弃抽象的“1-5分”或“A和B谁好”，转而建立基于具体维度的**二元 Checklist (Rubric)** 来累加得分。


有了 Evals 作为指南针，你在调整 Agent 的 Prompt 或工具链时将不再是“盲人摸象”。接下来，我们将深入探讨反思模式的最后一块拼图：**当反思机制能够获取到外部增量信息时，系统究竟会发生怎样令人惊叹的化学反应？**

### 使用外部反馈

在开发基于 LLM 的应用时，几乎每一个开发者都会经历一个痛苦的阶段：**提示词工程的边际效用递减**。

当你刚开始调整 Prompt 时（比如从零样本变为少样本），系统的性能（Performance）会有一个快速的拉升。但很快，这条曲线就会变得越来越平缓，直至完全停滞（Plateaus）。在这个阶段，无论你再怎么精心雕琢那些提示词，花上几个小时去替换一些形容词或调整语气，系统的整体表现都很难再有实质性的突破。

此时，**如果你感觉自己在无休止的“炼丹（Prompt Engineering）”中浪费时间，且收效甚微，那么是时候进行架构升维了。**

**1. 架构演进：从内部反思到外部闭环**

在这个停滞点，如果我们引入**基础的反思模式（Reflection）**——让 LLM 审视自己的初稿，系统的性能曲线会向上跃升一个台阶。但如果这种反思仅仅依赖模型内部的知识库（即“左手倒右手”），它依然会很快触达一个新的天花板。

**真正能让性能曲线呈现指数级突破、彻底改变游戏规则的，是引入“外部反馈（External Feedback）”。**

什么是外部反馈？它意味着在“初稿生成”和“反思修改”之间，插入一个**真实的工具执行层**。模型不再是凭空猜测哪里写得不好，而是基于物理或数字世界的真实反馈来进行迭代。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413140458814.png)

**2. 经典场景：外部反馈如何拯救 LLM 的短板？**

让我们看看在不同的业务挑战中，简单的外部工具是如何为反思机制注入灵魂的。

**(1) 代码沙盒 (Code Execution)**

如前文所述，当模型写出包含语法错误的代码（v1）时。仅靠模型自己看代码，很难发现隐藏的 Bug。但如果你通过代码执行器运行它，捕获到真实的报错日志 `SyntaxError: unterminated string literal (detected at line 1)`，并将其喂给模型。这种硬核的外部环境反馈，能让模型立刻定位问题并输出完美的 v2 版本。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413140624283.png)

**(2) 正则引擎匹配 (Pattern Matching)**

假设你在开发一个营销文案 Agent，严格禁止提及竞争对手。
*   **挑战**：LLM 偶尔会脑补出：“*我们的鞋子比 RivalCo 的好穿多了。*”
*   **外部反馈工具**：不需要用 AI 去查，直接写一段简单的 Python 正则表达式代码。
*   **反思闭环**：一旦代码检测到黑名单词汇，立即将触发词（“RivalCo”）作为确凿证据打回给 LLM，强迫其重写。

```python
# 伪代码：基于正则反馈的反思闭环
def marketing_agent(prompt):
    copy_v1 = llm.generate(prompt)
    
    # 外部工具：基于规则的硬性校验
    forbidden_words = regex_tool.find_competitors(copy_v1)
    
    if forbidden_words:
        # 将外部工具的确切反馈注入反思 Prompt
        reflection_prompt = f"Your draft contains forbidden competitor names: {forbidden_words}. Rewrite without them."
        return llm.generate(reflection_prompt)
        
    return copy_v1
```

**(3) 实时网络搜索 (Web Search)**

*   **挑战**：Agent 在写历史文章时声称“*泰姬陵建于 1648 年*”。但这其实是不严谨的（它是 1631 年动工，1648 年完工）。
*   **外部反馈工具**：调用 Web Search API 获取维基百科的权威摘要。
*   **反思闭环**：将检索到的真实历史片段喂给反思模型，模型便能基于最新、最准的外部事实，纠正自己的知识盲区，生成毫无破绽的叙述。


**(4) 精确字数统计 (Word Count Tool)**

* **挑战**：大家都知道，LLM 极度不擅长精确的字数控制（要求写 500 字，它经常写 600 字）。
* **外部反馈工具**：一个简单的 `len(text.split())` 函数。
* **反思闭环**：如果初稿字数超标，将真实的 `word_count` 数字反馈给 LLM：“*你的文章有 612 个字，超过了 500 字的限制，请大幅删减。*”有了具体的数字反馈，模型就能更精准地压缩篇幅。

  ![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413140901743.png)


**3. 结语与预告**

反思模式（Reflection），特别是结合了外部反馈的反思模式，是构建高可用 Agent 系统的第一块基石。它告诉我们一个核心的工程哲学：**不要强求大模型一次性完美，而是建立一个能让它不断试错、获取反馈并自我纠正的系统机制。**

我们在这节课中看到了许多“外部工具”的影子（如执行代码、网络搜索、正则匹配）。这正是我们通往更高级 Agentic 架构的桥梁。

在接下来的全新模块中，我们将正式深入探讨 Agentic AI 的第二大核心设计模式——**工具调用 (Tool Use)**。我们将学习如何系统性地赋予大模型调用外部函数的权利，让你的 Agent 彻底长出触角，连接广阔的真实世界！敬请期待！

## 工具调用

在前面的章节中，我们学习了如何通过任务拆解和反思机制（Reflection）来提升系统的文本质量。但随着应用场景的拓展，仅仅依赖大语言模型（LLM）内部预训练的知识体系，很快就会遇到无法逾越的边界。

正如人类在赤手空拳时能力极其有限，必须依靠锤子、扳手等工具才能改造世界一样；**如果想让 LLM 真正成为一个全能的 Agent，我们就必须赋予它使用工具（Tool Use）的能力。**

在 Agentic AI 的语境中，“工具”本质上就是由开发者编写、并允许 LLM 主动**请求调用**的计算机函数（Functions）。

**1. 为什么 LLM 需要工具？从一个简单的问题说起**

假设你问一个几个月前刚训练完成、完全离线的 LLM：“*现在是几点？*”
一个诚实且未产生幻觉的模型会回答：“*对不起，我没有获取当前时间的权限。*”

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414085429498.png)

因为大模型本质上是一个静态的神经网络权重集合，它被冻结在了训练完成的那一刻，根本不存在“当前时间”的概念。

为了解决这个问题，作为开发者的你，可以用 Python 写一个极简的函数：

```python
# 一个简单的工具函数：获取当前时间
from datetime import datetime

def get_current_time():
    """Returns the current time as a string."""
    return datetime.now().strftime("%H:%M:%S")
```

当你把这个函数作为“工具”注册给 LLM 后，奇妙的化学反应发生了：
1.  **用户提问**：“现在是几点？”
2.  **LLM 决策**：模型分析了意图，并意识到自己的知识库无法回答，但在它的可用工具列表（Tools）里，刚好有一个 `get_current_time`。
3.  **请求调用**：LLM 暂停文本生成，向系统发出请求：“请帮我执行 `get_current_time()`。”
4.  **环境执行与反馈**：系统执行该 Python 函数，得到字符串 `"15:20:45"`，并将其作为上下文悄悄传回给 LLM。
5.  **最终输出**：LLM 根据传回的时间信息，自信地回答：“*现在是下午 3:20。*”


**2. 核心机制：将控制权交还给 LLM**

在之前（比如反思模式的早期阶段）的硬编码设计中，我们作为工程师，会写死“*在哪一步必须进行 Web 搜索*”。

**但在真正的 Tool Use 机制下，工具的调用不是硬编码的，而是由 LLM 自主决策的。**

同样是配备了时间查询工具的 LLM，如果你问它：“*绿茶里通常含有多少咖啡因？*”
LLM 经过推理会发现：“回答这个问题根本不需要知道现在几点”。于是，它会直接跳过工具调用环节，利用内部知识库直接回答：“*一杯绿茶通常含有 25-50 毫克咖啡因...*”

**这种“按需调用（Choose tools when appropriate）”的自主性，正是 Agentic 架构最迷人的特质。**

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414085703573.png)

**3. 丰富的应用场景：打破物理与数字边界**

一旦理解了工具调用的逻辑，你就可以根据具体的业务需求，为模型打造专属的工具箱。

*   **本地生活与搜索**：
    *   *用户*：“帮我找找加州山景城附近的意大利餐厅？”
    *   *LLM 行为*：调用 `web_search(query="restaurants near Mountain View, CA")`，抓取最新营业信息。
*   **企业级数据问答 (Text-to-SQL)**：
    *   *用户*：“查一下买了白色墨镜的客户名单。”
    *   *LLM 行为*：调用 `query_database(table="sales", product="sunglasses", color="white")`，实现对私有资产的安全访问。
*   **精确的数学计算**：
    *   *用户*：“我存 500 美元，年利率 5%，10年后有多少钱？”
    *   *LLM 行为*：LLM 在做复杂算术时极易出错，但它可以调用专用的 `interest_calc(principal=500, interest_rate=5, years=10)`，或者更通用的代码执行引擎 `eval("500 * (1 + 0.05) ** 10")`，来确保结果 100% 精确。

    ![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414090016651.png)


**4. 进阶：多工具协作的工作流**

在更高级的场景中，我们往往会给 LLM 提供一整个工具矩阵（Multiple tools），让它通过组合调用来完成复杂的复合型任务。

以一个**智能日历助理 Agent** 为例。
*目标任务*：“*帮我看看周四日历上的空闲时间，并和 Alice 约个会。*”

为了完成这个目标，我们向模型注册了三个工具：`make_appointment`（预约）、`check_calendar`（查日程）、`delete_appointment`（取消预约）。

整个工作流将被 LLM 自主编排如下：

1.  **初次决策**：LLM 判断第一步需要知道周四的空闲状态，于是请求调用 `check_calendar("Thursday")`。
2.  **获取状态**：系统返回日历数据：“周四的 3pm, 4pm, 6pm 空闲”。
3.  **二次决策**：LLM 接收到空闲时间段后，决定挑选下午 3 点，并请求调用第二个工具 `make_appointment(person="Alice", time="Thursday 3pm")`。
4. **最终确认**：工具返回操作成功的确认码。LLM 最终用人类友好的语言回复用户：“*已经为您安排了周四下午 3 点与 Alice 的会议。*”

   ![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414090323378.png)


**5. 总结与展望**

赋予 LLM 工具调用的能力，是一次意义深远的架构升级。它让模型从一个“只懂纸上谈兵的学者”，进化成了一个“能联网查资料、能敲代码计算、能操作企业系统的全能数字员工”。

那么，在代码实现层面，这些“工具”究竟应该怎么写？我们如何向 LLM 描述一个函数的用途，确保它能准确无误地传递参数并触发调用？在接下来的章节中，我们将深入代码底层，手把手教你编写并注册属于你自己的 AI Tools。

### 工具调用的原理

初次接触 Agentic 工作流时，很多开发者都会感到困惑：**大语言模型（LLM）从本质上来说，只是一个被训练用来预测下一个单词（Token）的文本生成器。既然它只能输出文本，它究竟是怎么“调用”一段 Python 函数的？**

这是一个非常精彩的架构设计问题。为了彻底理解其中的奥秘，我们需要剥开现代高级 API 的封装，回到那个大模型还未被“原生训练”支持工具调用的早期阶段，去看看开发者最初是如何通过巧妙的提示词工程（Prompt Engineering）来实现这一壮举的。

**1. 本质：LLM 并不直接运行代码，它只是“请求”**

首先，我们需要明确一个核心概念：**LLM 绝对不会直接在你的服务器上执行任何代码。**
它所做的，仅仅是输出一段具有**特定格式的文本请求**。系统（也就是你写的调度代码）捕捉到这个请求后，代替 LLM 去执行本地函数，然后再把结果作为新的文本喂回给 LLM。

让我们通过一个简单的无参数工具——`get_current_time()`（获取当前时间），来还原这个完整的闭环。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414091508359.png)

**(1) 步骤一：在 System Prompt 中定义“暗号”**

由于早期的模型不懂什么是函数，我们需要在系统提示词（System Prompt）中和模型约定一个特定的“暗号”。

> **System Prompt**:
> *“你现在拥有一个名为 `get_current_time` 的工具。当且仅当你需要知道当前时间时，请停止其他回答，并严格输出以下精确格式的文本：*
> `FUNCTION:get_current_time()`”

**(2) 步骤二：大模型输出触发词**

当用户提问：“*现在是几点？*”
LLM 经过内部推理，发现自己无法直接回答，于是遵循 System Prompt 的指令，输出了我们约定的暗号：

> **LLM Output**: `FUNCTION:get_current_time()`

**(3)步骤三：开发者代码的拦截与执行**

此时，真正的魔法发生在**开发者编写的宿主调度代码**中。我们在获取 LLM 输出后，必须写一段逻辑来拦截这个暗号：

```python
# 伪代码：开发者拦截暗号并执行工具的过程

def agent_execution_loop(user_query):
    # 1. 发送用户问题给 LLM
    llm_output = llm.generate(user_query)
    
    # 2. 拦截暗号：检查输出中是否包含 "FUNCTION:"
    if "FUNCTION:" in llm_output:
        # 解析出函数名
        tool_name = extract_function_name(llm_output) # 这里提取到 'get_current_time'
        
        if tool_name == "get_current_time":
            # 3. 开发者代码代替 LLM 真正执行 Python 函数
            result = get_current_time() # 假设返回 "08:00:00"
            
            # 4. 将结果追加到对话历史中，再次喂给 LLM
            conversation_history.append(f"Tool executed successfully. Result: {result}")
            final_answer = llm.generate(conversation_history)
            
            return final_answer
            
    return llm_output
```

最终，LLM 接收到了系统传回的 `"08:00:00"`，并自然地输出最终答复：“*现在是早上 8 点。*”

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414091758944.png)

**2. 进阶：如何处理带参数的函数？**

如果工具变得复杂，需要传入参数怎么办？流程是完全一样的，只是我们的“暗号”格式需要升级。

假设我们升级了时间查询工具，允许查询特定时区的时间：

```python
from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time(timezone: str):
    """Returns current time for the given time zone (e.g., 'Pacific/Auckland')."""
    tz = ZoneInfo(timezone)
    return datetime.now(tz).strftime("%H:%M:%S")
```

此时，我们需要修改 System Prompt，告诉模型不仅要输出暗号，还要把参数带上：
> *“要使用该工具，请输出格式：`FUNCTION:get_current_time("指定的时区")`”*

当用户问：“*新西兰现在几点？*”
LLM 凭借其世界知识，知道新西兰属于奥克兰时区，于是输出：
> **LLM Output**: `FUNCTION:get_current_time("Pacific/Auckland")`

开发者代码再次拦截到这串文本，提取出参数 `"Pacific/Auckland"`，将其传入本地函数执行，得到结果（如 `"04:00:00"`），再喂给 LLM，最终完成闭环。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414100533237.png)

**3. 总结：从“笨拙的正则匹配”到“现代原生 API”**

总结一下，让 LLM 使用工具的底层生命周期分为四步：
1.  **供给 (Provide)**：实现函数，并在 Prompt 中告诉 LLM 其存在与用法。
2.  **请求 (Request)**：LLM 根据需求，输出特定格式的调用请求。
3.  **执行 (Execute)**：外部代码拦截请求，解析参数，并在本地执行真实函数。
4.  **反馈 (Feedback)**：将函数返回值喂回给 LLM，让其继续下一步推理或输出最终结果。


这种依赖“大写字母 FUNCTION”和字符串解析（正则匹配）的方式，虽然能跑通，但显得非常笨拙（Clunky），且容错率低（如果模型忘记打括号，代码就会崩溃）。

好消息是，随着技术的演进，今天的顶尖大模型（如 GPT-4、Claude 3）在预训练或微调阶段，已经被**原生训练**了如何使用工具。我们不再需要手写那些蹩脚的“暗号解析”代码了。

那么，在现代的 Agentic 框架中，标准的、优雅的工具注册与调用语法究竟长什么样？在下一节中，我们将带你走进现代 LLM Function Calling 的世界。

### 工具调用的语法

在上一节中，我们探讨了早期让 LLM 使用工具的“笨办法”——通过在 System Prompt 中约定大写字母（如 `FUNCTION:get_current_time()`）来触发调用，并由开发者手动编写正则拦截和执行代码。

这种方式既繁琐又容易出错。幸运的是，随着大语言模型生态的进化，现代顶尖大模型（如 GPT-4o、Claude 3.5）不仅在预训练时原生支持了工具调用（Function Calling），各大中间件库（如 OpenAI SDK、AISuite 等）也为开发者封装了极其优雅的语法。

今天，我们将通过 `aisuite` 这个强大的开源多模型适配库，深入探讨现代 Agentic 开发中工具调用的代码骨架以及其背后的运行机制。

*(注：在行业术语中，虽然严格来说是“LLM 请求系统调用工具”，但为了交流方便，开发者们通常会直接简称为“LLM 调用了工具”。)*

**1. 现代优雅的工具注册语法**

让我们再次请出那个简单的无参数工具：`get_current_time()`。

在现代开发中，你不需要再写又长又臭的 System Prompt 去解释这个工具怎么用了。你只需要保持良好的 Python 编程习惯，为函数写好**文档字符串（Docstring）**即可：

```python
from datetime import datetime
import aisuite as ai

# 1. 定义工具函数，必须包含清晰的 Docstring
def get_current_time():
    """Returns the current time as a string"""
    return datetime.now().strftime("%H:%M:%S")

# 2. 初始化客户端
client = ai.Client()

# 3. 构造请求，直接将函数名传入 tools 列表
response = client.chat.completions.create(
    model="openai:gpt-4o",  # 使用 OpenAI 的 GPT-4o 模型
    messages=messages,      # 用户的对话历史
    tools=[get_current_time], # 神奇之处：直接传入 Python 函数对象
    max_turns=5             # 保护机制：防止 LLM 陷入无限调用死循环
)
```

**代码解析：**
这套语法与标准的 OpenAI SDK 非常相似。核心魔法在于 `tools=[get_current_time]` 这一行。
你可能会好奇：**LLM 是怎么知道这个函数干嘛用的？**

这就引出了底层转换机制：中间件库（如 `aisuite`）在发送请求前，会使用反射机制（Reflection in Python）自动读取这个函数的函数名和它的 `Docstring`，并将其翻译成 LLM 能够理解的标准格式。

*（关于 `max_turns`：当一个工具执行完毕后，LLM 可能会根据返回结果决定调用下一个工具。`max_turns=5` 就是设置一个天花板，允许模型最多连续请求调用 5 次工具，防止它陷入逻辑死循环。在正常业务中极少会触发这个上限。）*

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414113140423.png)

**2. 揭秘底层原理：自动生成的 JSON Schema**

为了不让这一切显得像黑魔法，我们来看看在你调用 `client.chat.completions.create` 时，底层到底发送了什么给大模型。

实际上，大模型需要的是一个严格结构化的 **JSON Schema**。对于上面的无参数函数，底层自动生成的 JSON Schema 如下：

```json
// 无参数工具的 JSON Schema 描述
"tools": [
  {
    "type": "function",
    "function": {
      "name": "get_current_time",
      "description": "Returns the current time as a string",
      "parameters": {} 
    }
  }
]
```
你看，函数的 `name` 和 `description` 被精准地提取并注入到了 JSON 中。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414113352147.png)

**3. 进阶：带参数函数的 Schema 生成**

如果是带有参数的复杂函数呢？比如我们上一节提到的带时区的查询：

```python
from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time(timezone: str):
    """
    Returns current time for the given time zone.
    
    Parameters:
        timezone (str): The IANA time zone string, e.g., 'America/New_York' or 'Pacific/Auckland'.
    """
    timezone = ZoneInfo(timezone)
    return datetime.now(timezone).strftime("%H:%M:%S")
```

中间件库会解析函数的签名（类型提示 `str`）以及 Docstring 中对参数的详细描述，构建出更加复杂的 JSON Schema：

```json
// 带参数工具的 JSON Schema 描述
"tools": [
  {
    "type": "function",
    "function": {
      "name": "get_current_time",
      "description": "Returns the current time for the given timezone.",
      "parameters": {
        "type": "object",
        "properties": {
          "timezone": {
            "type": "string",
            "description": "The IANA time zone string, e.g., 'America/New_York' or 'Pacific/Auckland'."
          }
        },
        "required": ["timezone"]
      }
    }
  }
]
```

当模型收到这份极其规范的“工具使用说明书”后，如果用户问“*新西兰现在几点？*”，模型就能极其精准地输出一个 JSON 指令，要求调用 `get_current_time` 并传入参数 `"Pacific/Auckland"`。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414122937580.png)

**4. 现代框架的全自动执行闭环**

在原生 API 开发中，当 LLM 返回包含工具调用请求的 JSON 时，开发者依然需要手动拦截、执行函数，并将结果拼接到 Message 中再发回给 LLM。

**而现代高级封装库（如本例中的 `aisuite`）帮你把“执行”这一步也包揽了。**

当你在 `tools` 参数中传入了真正的 Python 函数对象时，一旦 LLM 发起调用请求，`aisuite` 客户端会自动在本地运行该函数，获取返回值，喂给 LLM，并在一个单向的函数调用中自动完成多次对话轮转（直到得到最终文本结果或达到 `max_turns`）。

**5. 总结与展望**

通过良好的 Docstring 编写习惯和现代化的 SDK 封装，向大模型赋予工具（Tool Use）的开发体验已经变得极其丝滑。在接下来的实战 Lab 中，当你第一次看到几行简单的代码就能让大模型自主决定去搜索网络、查询数据库时，那种“系统真正活过来”的震撼感是无与伦比的。

在你可以赋予大模型的所有工具中，有一个工具具有最顶级的特权地位：**代码执行器 (Code Execution)**。

如果你给 LLM 自由编写和运行代码的权利，它就相当于拥有了无限的工具。在下一节中，我们将深入探讨这个堪称 Agentic 架构“核武器”的特殊工具。

### 代码执行器

在众多赋予大语言模型（LLM）的工具中，有一种工具拥有着至高无上的特权地位。当你在 Agentic 工作流中为模型接入这个工具时，你经常会被它生成的巧妙解决方案所震撼。

这个工具就是：**代码执行器（Code Execution）**。

为什么它如此特殊？因为代码是图灵完备的。当你赋予 LLM 编写并运行代码的能力时，你实际上是赋予了它“制造无限工具”的能力。

**1. 从“无尽的工具堆砌”到“动态代码生成”**

为了理解代码执行器的价值，我们先来看一个反面教材：**如何用 LLM 构建一个数学计算器？**

如果你采用常规的“按需提供工具”的思路，你可能会先写四个工具：`add`（加）、`subtract`（减）、`multiply`（乘）、`divide`（除）。
当用户问：“*13.2 加 18.9 等于多少？*” LLM 顺利调用了 `add` 工具。

但紧接着，用户问：“*2 的平方根是多少？*”
这时你傻眼了。难道你要再跑去写一个 `square_root` 的工具函数吗？如果你看一下现代科学计算器上密密麻麻的按键，你会发现为每一种数学运算都预先硬编码一个工具，是完全不切实际的。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414125540906.png)

**更优雅、更具扩展性的替代方案是：让模型现场写代码来计算。**

我们可以通过一段 System Prompt 来改变模型的工作模式：
> *“Write code to solve the user's query. Return your answer as python code delimited with `<execute_python>` and `</execute_python>` tags.”*
> *(“编写代码来解决用户的查询。请将你的答案作为 Python 代码返回，并使用 `<execute_python>` 标签包裹。”)*

当用户再次问“*2 的平方根是多少？*”时，工作流将产生极其精妙的变化：

1.  **LLM 生成代码**：模型直接输出如下文本：
    ```python
    <execute_python>
    import math
    print(math.sqrt(2))
    </execute_python>
    ```
2.  **提取与执行**：宿主程序通过正则表达式提取标签内的 Python 代码，并调用解释器（如 Python 原生的 `exec()` 函数）来运行这段代码。
3.  **捕获与反馈**：解释器吐出结果 `1.4142135623730951`，并将其传回给 LLM。
4.  **最终输出**：LLM 给出人性化的回答：“*2 的平方根大约是 1.4142。*”


通过这种方式，不管是算复利、解微积分还是做复杂的统计学回归分析，只要是 Python 代码能解决的计算问题，LLM 都能游刃有余地处理。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414124458294.png)

**2. 叠加“反思闭环”：让代码执行更加坚不可摧**

代码执行最强大的地方在于，它可以与我们之前学过的**反思机制（Reflection with external feedback）**完美结合。

LLM 写的代码并不总是完美的。假设系统在执行模型生成的初稿代码（v1）时，触发了 `SyntaxError`（语法错误）或 `TypeError`。在传统的应用中，系统直接就崩溃了。

但在配备了代码执行器的 Agentic 架构中，系统会捕获这些错误日志（Error logs），并将它们作为外部反馈重新喂给 LLM：
*“你写的这段代码报错了，报错信息是：[Error log]... 请反思并修改。”*

模型会迅速定位 Bug，输出修正后的第二版代码（v2）。这种“写代码 -> 运行报错 -> 修改代码 -> 运行成功”的闭环，正是现代高级智能体（如 OpenAI 的 Advanced Data Analysis）的底层运行逻辑。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414125655302.png)

**3. 安全警钟：千万不要在宿主环境裸奔 (Secure Code Execution)**

看到这里，有经验的后端工程师可能已经惊出一身冷汗了。

**执行大模型生成的任意代码（Arbitrary code），具有极高的系统风险。** 哪怕你用的是最聪明的模型。

吴恩达（Andrew Ng）教授的团队就曾遇到过一个真实的灾难案例：一个高度自主的 Coder Agent 在执行任务时，由于逻辑判断失误，竟然在项目目录下执行了 `rm *.py`（删除所有 Python 文件）的毁灭性操作。
事后，当团队将错误日志喂给模型让其反思时，这个 AI 还极度拟人化地道了歉：*“是的，您完全正确——那是一个极其愚蠢的错误。我绝对不该在项目目录中使用 rm *.py。”*
虽然它的态度很诚恳，但如果没有版本控制（GitHub），几个月的代码心血可能就化为乌有了。

因此，**将 LLM 生成的代码放在沙盒（Sandbox）外运行，是绝对的工程禁忌。**

为了防止数据丢失、敏感信息泄露或恶意系统级攻击，最佳实践是将代码调度到极其受限的隔离环境中执行。目前业界常用的安全方案包括：
*   **Docker 容器化**：将代码扔进没有网络访问权限、挂载只读盘的隔离容器中运行。
*   **E2B (e2b.dev)** 等现代 AI 沙盒云服务：提供毫秒级启动的轻量化微型虚拟机，专门用于安全执行 Agent 生成的代码。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414125149483.png)

**4. 总结与展望**

代码执行器（Code Execution）之所以被称为大模型的“核武器”，是因为它彻底打破了“预设工具”的局限，让 AI 拥有了现场制造工具、处理无限复杂问题的能力。正因其如此重要，当今几乎所有的顶尖大模型都在预训练阶段接受了大量的代码语料训练，以确保它们能够写出高质量的执行流。

然而，在业务开发中，除了让大模型自己写代码，我们依然需要频繁地接入大量的第三方工具（如查天气、查股票、调用企业内部 CRM 等）。如果每个开发者都要自己去查 API 文档、写工具封装代码，那将是巨大的重复造轮子。

有没有一种标准协议，能够让大模型一键接入全世界的 API 工具呢？

在下一节中，我们将向你介绍一个正在席卷 AI 圈的全新工业级标准——**MCP (Model Context Protocol)**。它正致力于改变开发者为 LLM 提供工具的底层生态。敬请期待！

### MCP

当你掌握了如何通过函数声明向大模型提供工具后，随之而来的现实问题是：**在真实世界的商业开发中，你需要接入的工具数量是海量的。**

你可能希望你的 Agent 能读取 Slack 上的聊天记录、检索 Google Drive 里的文档、拉取 GitHub 的代码 PR，甚至查询企业内部部署的 PostgreSQL 数据库。

在过去，这对于开发者来说简直是一场噩梦。

**1. 传统的痛点：M × N 的开发地狱**

假设有 `M` 个独立的 AI 应用（App 1, App 2, App 3...），市面上有 `N` 种常用的第三方工具或数据源（Slack, GDrive, Github...）。

在没有统一标准的情况下，如果开发者想让自己的 App 接入 Slack，他必须去研读 Slack 的 API 文档，自己写一套 Wrapper（封装层），把 Slack 的接口翻译成 LLM 能够理解的 JSON Schema（即我们在前面提到的 Tool Syntax）。
如果另一个团队开发了 App 2，他们又得从头把 Slack 重新封装一遍。

这就导致了整个开源和开发者社区在做极其低效的重复劳动。总的开发工作量是 **M × N**。大家都在忙着造各种五花八门的“数据源适配器”。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414132026292.png)

**2. 破局：MCP 架构的 M + N 革命**

为了终结这种混乱，AI 巨头 Anthropic（Claude 系列模型的母公司）提出了一个极具野心的开源标准——**Model Context Protocol (MCP，模型上下文协议)**。目前，这一标准已经被业界各大公司和广大开发者迅速采纳。

MCP 的核心思想非常简单：**在 AI 应用（Client）和数据源工具（Server）之间，架设一层统一的标准化通信协议。**

在 MCP 的架构下：
*   **各大 SaaS 厂商和开源社区（Servers）**：只需要按照 MCP 标准，把自己的 API 封装成一个 **MCP Server**（例如，GitHub 官方提供一个 GitHub MCP Server）。
*   **各大 AI 应用开发者（Clients）**：只需要让自己的应用支持作为 **MCP Client** 接入。


这样一来，总的开发工作量骤降为 **M + N**。只要你的 AI App 接入了 MCP 协议，你就瞬间拥有了调用全世界所有已开源 MCP Server 的能力，无需再写哪怕一行底层的 API 封装代码！

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414132220389.png)

**3. MCP 的核心设计：获取上下文资源与调用工具**

MCP 最初的设计初衷，是为了解决 LLM 获取上下文（Context）困难的问题。因此，在 MCP 的官方文档中，很多只读的数据抓取工具被称为 **Resources（资源）**。但随着协议的演进，MCP 同样完美支持了具有副作用的 **Functions / Tools（功能调用）**。

为了更直观地理解 MCP 的威力，我们来看一个具体的实战案例。

**场景：使用 Claude Desktop（一个 MCP Client）分析 GitHub 仓库。**

假设你在本地运行了 Claude Desktop 客户端，并在配置中连接了由开源社区维护的 **GitHub MCP Server**。

**对话轮次 1：获取上下文 (Resources)**
*   **你**：“*请帮我总结一下这个 GitHub 仓库（某 AI 项目的 URL）下的 readme.md 文件。*”
*   **MCP 运转**：客户端并没有去写爬虫，而是直接向 GitHub MCP Server 发送了一个标准请求：“*请获取该 repo 下的 readme.md 文件。*” Server 返回大段的文件内容。
*   **LLM 处理**：这篇长文本被无缝注入到 LLM 的上下文窗口中，LLM 顺利输出了总结报告。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414132453639.png)

**对话轮次 2：执行高级工具查询 (Tools)**

*   **你**：“*列出这个仓库最近的 Pull Requests (PR)。*”
*   **MCP 运转**：客户端再次调用 MCP Server 提供的专有工具 `list_pull_requests`，并带上了 `repo_name` 和 `limit=20` 等参数。
*   **LLM 处理**：Server 执行查询并返回结构化的 PR 列表数据，LLM 基于这些数据，为你生成了一份包含最近更新内容的优雅文本摘要。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260414133046374.png)

**4. 加入蓬勃发展的 MCP 生态**

今天，MCP 的生态正在以惊人的速度扩张。

*   **知名的 MCP Clients**：除了 Claude Desktop，目前极其火爆的 AI 编程 IDE，如 **Cursor** 和 **Windsurf**，都已经深度集成了 MCP。这意味着你可以直接在 IDE 里让 AI 帮你查阅外部数据库的表结构。
*   **海量的 MCP Servers**：不仅有 Slack、Google Drive、PostgreSQL 的现成 Server，还有无数开发者正在为各种小众工具编写 MCP Server 接入点。


对于今天的 Agent 开发者而言：

1.  **你的 AI 应用**应该致力于成为一个标准的 **MCP Client**，从而白嫖整个开源生态的工具箱。
2.  如果你开发了一个企业内部独有的系统（比如你公司的私有 OA 系统），你可以花半天时间将它封装成一个 **MCP Server**。这样，你公司内部的各种 AI 助手就能无缝地调用该 OA 系统的数据了。

**5. 结语与预告**

掌握了**任务拆解、反思机制（Reflection）**以及强大的**工具调用体系（Tool Use & MCP）**，你已经跨入了高级 Agentic 架构师的门槛。

但是，拥有了这些强大的武器，你构建出来的 Agent 就一定好用吗？当你把大模型、多步思考、外部工具全部耦合在一起时，整个系统会变得极度脆弱。一次错误的工具调用，就可能导致最终的回答南辕北辙。

正如吴恩达教授所言：**区分顶尖 Agent 团队与平庸团队的最核心指标，在于他们是否能够建立一套严谨的“评估与错误分析流程”。**

在下一个模块中，我们将迎来本系列课程中最具工程价值、也最为重要的一环：**如何利用 Evals（评估）与 Error Analysis（错误分析）来驱动 Agent 系统的持续迭代。** 我们下个模块见！

## 构建 Agentic AI 的实用技巧

在构建 Agentic AI 工作流时，开发者往往会面临一个巨大的困境：**在系统真正跑起来之前，你根本无法预知大模型会在哪里“翻车”。**

与其花上几周的时间在白板上疯狂推演、试图穷举所有可能的错误，不如采用一种更敏捷的工程哲学：**快速搭建一个安全的初始原型（Quick and dirty system），直接观察它的真实输出，让系统的“失败”来指导你构建评估体系（Evals）。**

当你通过人工抽检（比如查看 20 个运行结果）发现了系统的软肋后，接下来要做的，就是针对这个具体的软肋，编写自动化的 Eval 脚本，以此来驱动后续的迭代。

根据业务场景的不同，构建 Eval 的方式大体可以分为以下三种核心模式。

**1. 模式一：带有明确“标准答案”的客观评估 (Ground Truth Evals)**

**实战场景：自动化发票处理 (Invoice Processing)**

假设你的 Agent 需要从复杂的 PDF 发票中提取 4 个核心字段（开票方、地址、金额、到期日），并写入数据库。

当你抽检了 20 张发票的处理结果后，你发现了一个高频错误：大模型经常把“发票开具日（Invoice Date）”和“付款到期日（Due Date）”搞混。这是一个致命的业务错误，因为搞错到期日会导致企业违约。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415085117807.png)

此时，你需要建立一个专门针对“到期日提取准确率”的 Eval：
1. **人工标注（Ground Truth）**：找 20 张测试发票，人工记录下正确的到期日。
2. **规范输出格式**：修改系统提示词，强制要求模型将日期输出为极易被代码解析的格式（如 `YYYY/MM/DD`）。
3. **编写评估代码**：使用正则表达式截取大模型输出的日期，并与人工标注的标准答案进行严格比对。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415090204329.png)

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415090222995.png)

```python
# 模式一代码示例：基于正则匹配的客观评估
import re

def evaluate_invoice_due_date(llm_response, actual_date_ground_truth):
    """
    评估发票到期日的提取准确率
    """
    # 定义严格的日期匹配模式 (YYYY/MM/DD)
    date_pattern = r'\d{4}/\d{2}/\d{2}'
    
    # 从 LLM 输出的文本中提取日期
    extracted_dates = re.findall(date_pattern, llm_response)
    
    if not extracted_dates:
        return False # 提取失败
        
    extracted_date = extracted_dates[0]
    
    # 将提取结果与我们人工标注的 Ground Truth 进行对比
    if extracted_date == actual_date_ground_truth:
        return True
    return False

# 迭代目标：通过修改 Prompt，让 num_correct / 20 的比例不断逼近 100%
```

**2. 模式二：无标准答案，但有“硬性规则”的客观评估 (Rule-based Evals)**

**实战场景：Instagram 营销文案助手 (Marketing Copy Assistant)**

假设你开发了一个多模态 Agent：输入一张产品图（比如太阳镜），让模型生成一条用于发在 Instagram 上的营销文案。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415090431854.png)营销团队给出了一个硬性规定：**文案长度绝对不能超过 10 个单词。**

当你抽检输出时发现，模型虽然写的文案很吸引人，但经常无视长度限制（比如写了 17 个词或 14 个词）。

构建这种 Eval 的特点是：**它不需要逐一标注“标准答案”（因为文案怎么写都行），但必须满足全局的硬性规则。**

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415090758199.png)

```python
# 模式二代码示例：基于代码规则的客观评估

def evaluate_marketing_copy_length(generated_text, limit=10):
    """
    评估生成的营销文案是否符合长度限制
    """
    # 简单的按空格分词统计词数
    word_count = len(generated_text.split())
    
    # 只要字数达标，该测试用例即算通过
    if word_count <= limit:
        return True
    else:
        return False
        
# 迭代目标：测试 20 张不同的商品图，要求生成文案的长度合规率达到 100%
```

**3. 模式三：利用大模型作为裁判的主观评估 (LLM-as-a-Judge Evals)**

**实战场景：深度研究智能体 (Research Agent)**

随着任务变得复杂，代码逻辑（正则匹配、算长度）就不够用了。假设你让 Agent 撰写关于“黑洞科学的最新进展”或“用于采摘水果的机器人技术”的深度研究文章。

抽检结果显示：文章文笔优美、没有语法错误，**但经常遗漏行业专家都会关注的核心论点（High-profile results）**。比如在写水果采摘机器人时，完全没提行业内最顶尖的那家设备公司。

面对这种主观且复杂的“内容深度”问题，我们需要引入 **LLM-as-a-Judge（让大模型当裁判）** 的评估模式：
1. **提炼黄金标准（Gold-standard points）**：针对每个测试课题，人工写出 3-5 个必须提及的核心知识点（例如针对黑洞，必须提及“事件视界”和“射电望远镜”）。
2. **构建裁判 Prompt**：将 Agent 生成的文章和我们定的“黄金标准”一起喂给一个强大的裁判模型。
3. **强制结构化输出**：要求裁判模型输出 JSON，以便于我们的工程代码解析评分。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415091021516.png)

```python
# 模式三代码示例：调用 LLM 裁判进行打分与解析
import json

def evaluate_research_depth_with_llm(original_prompt, essay_text, gold_standard_points):
    """
    使用 LLM 裁判评估文章是否覆盖了核心知识点
    """
    # 构造极其严谨的裁判指令
    judge_prompt = f"""
    Determine how many of the gold-standard talking points are present in the provided essay.
    
    Original Prompt: {original_prompt}
    Essay to Evaluate: {essay_text}
    Gold Standard Talking Points: {gold_standard_points}
    
    Output Format:
    Return a JSON object with strictly two keys:
    - 'score': a single integer representing the number of points covered.
    - 'explanation': a string briefly explaining which points were found.
    """
    
    # 调用大模型 (此处省略底层调用逻辑)
    judge_response_json_str = llm_judge.generate(judge_prompt)
    
    # 解析 JSON 获取客观分数
    result = json.loads(judge_response_json_str)
    
    return result["score"], result["explanation"]

# 迭代目标：在包含 20 个课题的测试集中，让每篇文章的平均得分尽可能接近满分（如 5 分）。
```

**4. 小结：用 Evals 驱动 Agent 的进化**

通过以上三个场景，我们可以清晰地看到高级 Agent 工程师的日常工作流：**发现缺陷 -> 构建包含 10~20 个样本的初始 Eval 集 -> 编写评估脚本 -> 修改系统架构/Prompt -> 观察得分是否上升。**

随着系统的成熟，如果你发现这 20 个样本已经无法覆盖新的业务场景，你可以随时向评估集中添加新的测试用例（Test cases）。这就是软件工程中经典的“测试驱动开发（TDD）”在 AI 时代的完美复刻。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415091351492.png)

### 评估策略矩阵

刚才，我们通过实际的代码案例，展示了三种不同类型的端到端评估（End-to-End Evals）方法。

但在真实的复杂架构中，面对千变万化的业务需求，我们应该如何系统性地选择最合适的评估策略？吴恩达（Andrew Ng）教授为我们提供了一个极具威力的思维模型：**评估策略的 2x2 矩阵**。

**1. Evals 的两个核心坐标轴**

构建评估体系时，我们可以沿着两个核心“坐标轴（Axes）”来进行分类决策。

**横轴：评估手段（Evaluation Method）**
*   **代码驱动（Objective）**：非黑即白，通过编写 Python 逻辑（如正则匹配、字符串对比、长度判断）来给出确定性的通过或失败。
*   **大模型裁判（LLM-as-a-judge / Subjective）**：用于处理具有主观性或复杂语义表达的任务，利用 LLM 的推理能力给出评分或判断。

**纵轴：基准数据的存在形式（Ground Truth Type）**
*   **按例独立标注（Per-example Ground Truth）**：测试集中的每一个样本，都有一个独一无二的“标准答案”（比如特定日期的字符串、特定知识点的集合）。
*   **无特定答案（No per-example Ground Truth）**：测试集中没有具体的答案，而是依靠全局通用的“硬性准则”来进行约束。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415092305231.png)

将这两个坐标轴交叉，我们就得到了涵盖几乎所有场景的 **2x2 评估矩阵**：

1.  **代码驱动 + 独立标注 (Objective & Per-example)**：
    *   *典型案例*：发票到期日提取。
    *   *逻辑*：`if extracted_date == actual_date: num_correct += 1`（硬编码对比具体日期）。
2.  **代码驱动 + 无特定答案 (Objective & No per-example)**：
    *   *典型案例*：Instagram 营销文案助手。
    *   *逻辑*：`if len(text) <= 10: num_correct += 1`（文案随便写，只要字数不超全局限制即可通过）。
3.  **模型裁判 + 独立标注 (Subjective & Per-example)**：
    *   *典型案例*：研究报告的知识点覆盖度。
    *   *逻辑*：通过 LLM 裁判，判断生成的文章是否包含了我们针对该主题预先设定的“特定核心论点”。
4.  **模型裁判 + 无特定答案 (Subjective & No per-example)**：
    *   *典型案例*：基于量表（Rubric）给生成的图表打分。
    *   *逻辑*：没有预设的“标准图表”，而是利用 LLM 裁判，基于一套全局统一的审美规范（如：坐标轴是否清晰、标题是否存在）对任何图表进行打分。

熟练掌握这个 2x2 矩阵，可以让你在接手任何 Agent 项目时，瞬间定位出最适合的评估策略，从而避免走弯路。

**2. 敏捷 Evals 的三大实战法则**

在指导了大量 AI 团队之后，行业专家发现了一个普遍的现象：许多团队陷入了**“评估瘫痪（Evaluation Paralysis）”**。
他们认为建立一套 Evals 是一项耗时数周的庞大工程，必须做到完美无缺才敢上线。结果就是迟迟不敢动手。

为了打破这种僵局，请牢记以下三条高级工程法则：

**法则一：Quick and dirty is ok to start!（先让它跑起来！）**
构建 Evals 的过程，和开发 Agent 工作流本身一样，是一个持续迭代的过程。
不要追求完美的测试集。先手写 15 到 20 个测试样例，写几行简单的判定代码，或者写个简单的裁判 Prompt。这就足以给你一个基准分数（Baseline metric）。这个分数结合你人工审阅的直觉，就能立刻驱动你做出下一步的优化决策。

**法则二：用“直觉背离”来升级指标体系**
随着系统不断迭代，你会遇到一种典型的困境：*你修改了底层逻辑，人工抽检发现系统明明变聪明了，但 Eval 跑出来的分数反而下降了。*
不要慌张，这正是你的系统进化的标志！这说明你早期的 Eval 脚本已经无法捕捉人类的高级判断（Fail to capture human judgement）。这就是你扩充测试集、或者细化量表（Rubric）规则的绝佳时机。Evals 必须随着系统的聪明程度而同步升级。

**法则三：将“人类专家”作为寻找优化灵感的基准**
对于绝大多数旨在实现流程自动化的 Agent 工作流，寻找突破口最简单的方法就是：**找到那些 Agent 表现还不如人类专家的地方。**
如果人类能轻易写出涵盖所有痛点的文章，而 Agent 总是在某些点上栽跟头，那么这些具体的“失败案例”，就是你下一步重点优化和构建 Eval 测试用例的宝贵灵感来源。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415092642814.png)

**3. 结语与预告**

当你构建好一套粗糙但有效的端到端评估（End-to-End Evals）系统后，你终于有了量化系统表现的准绳。

然而，现代 Agentic 系统往往是由多个复杂的组件（如 Planner、Retriever、Coder）串联而成的。当端到端的 Eval 告诉你“系统只拿了 60 分”时，**你该如何知道到底是哪一个组件拖了后腿？你有限的优化精力到底该投入到哪个环节？**

在下一节中，我们将深入 Agent 的架构内部，探讨一项极其核心的高阶能力：如何通过**错误分析（Error Analysis）**和**组件级评估**，对庞大的系统进行精准把脉与调优。敬请期待！

### 错误分析与系统归因

当你满怀激情地构建好一个 Agentic 工作流，并迫不及待地让它跑起来时，现实往往会给你浇一盆冷水——**系统初期的表现大概率是不尽如人意的。**

这太正常了。问题不在于系统“翻车”，而在于**翻车之后，你应该把有限的精力投入到哪里？**

一个复杂的 Agentic 系统由无数个精密咬合的齿轮组成（如 Prompt 生成、工具调用、大模型推理）。当系统输出了一篇低质量的研究报告时，很多开发者会习惯性地“拍脑袋”决策：“*我觉得是大模型能力不够，我换个 Prompt 试试*”，或者“*肯定是这个大模型变笨了，我切到 GPT-4 看看*”。

这种**凭直觉（Go by gut）**的盲目调优，往往会导致团队在某个并不致命的组件上耗费数周时间，最终系统整体性能毫无起色。

区分顶尖 Agent 架构师与普通开发者的核心标志，就在于前者拥有一套极其严谨的**错误分析（Error Analysis）与系统归因流程**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415123257017.png)

**1. 拆解黑盒：追踪 (Traces) 与跨度 (Spans)**

为了进行错误分析，我们必须把 Agent 视作一个透明的流水线，仔细检查水流经过每一个节点的纯度。

在系统可观测性（Observability）领域，我们借用两个核心术语：
*   **Trace（追踪）**：一次完整的用户请求从输入到最终输出的整个生命周期记录。
*   **Span（跨度）**：生命周期中单个独立步骤（如一次特定的 Web 搜索，或一次特定的 LLM 生成）的局部输出。

假设我们的**“黑洞深度研究 Agent”**最终生成的文章**“遗漏了核心的科学论点（Missed key points）”**。我们应该如何顺藤摸瓜？

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415123521241.png)

我们需要调出这次执行的完整 Trace，并逐个 Span 进行审查：

1.  **Span 1: 生成搜索词 (LLM)**
    *   *输出检查*：`"Black hole theories Einstein"`, `"Event horizon telescope radio"`。
    *   *评估*：人类专家认为这些搜索词很精准，没有问题。
2.  **Span 2: 执行 Web 搜索 (Search Engine)**
    *   *输出检查*：返回的 URL 列表中包含了大量类似 *“小学生破解 30 年黑洞谜题 (AstroKidNews)”* 或个人博客的文章。
    *   *评估*：**发现严重问题！** 搜索引擎返回的学术严谨度极低。
3.  **Span 3: 筛选 Top 5 信息源 (LLM)**
    *   *输出检查*：LLM 在一堆垃圾文章里，挑出了它认为最相关的 5 篇。
    *   *评估*：虽然最终挑出的文章很差，但**我们不能把锅甩给这一步的 LLM**。因为上游传下来的数据全是垃圾，“巧妇难为无米之炊”，这个组件在当时的环境下已经尽力了。

通过这样逐个 Span 的拆解，我们精准地定位到：**罪魁祸首是 Span 2 的搜索引擎调用逻辑**（也许是使用的默认搜索引擎太偏向大众娱乐，我们需要切换到 Google Scholar 或调整搜索参数）。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415124012117.png)

**2. 科学归因：建立错误统计电子表格**

当然，仅凭一次 Trace 发现的问题可能是偶然的。为了获得宏观的决策依据，我们需要聚焦于那些**“最终输出不合格（Errors）”**的测试用例，并建立一个追踪电子表格（Spreadsheet）进行量化统计。

对于每一个失败的用例，人类专家需要逐一审查其 Trace，并在“表现明显低于人类水准”的 Span 下打上标记。

统计结果可能如下：

| 失败用例课题           | 搜索词生成 (Search terms) | 搜索引擎返回结果 (Search results) | 信息源筛选 (Picking sources)       | ...  |
| :--------------------- | :------------------------ | :-------------------------------- | :--------------------------------- | :--- |
| **黑洞最新进展**       | (表现正常)                | ❌ 包含过多大众博客，缺乏论文      | (上游数据导致，不计入当前组件错误) |      |
| **西雅图租房 vs 买房** | (表现正常)                | (表现正常)                        | ❌ 漏掉了西雅图最权威的房地产博客   |      |
| **水果采摘机器人**     | ❌ 搜索词过于宽泛          | (上游数据导致)                    | (上游数据导致)                     |      |
| ...                    | ...                       | ...                               | ...                                |      |
| **总体错误占比**       | **5%**                    | **45%**                           | **10%**                            |      |

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415124437735.png)

**3. 数据驱动的优化决策**

当这张表格摆在你面前时，接下来的优化路径就不再是盲目的了。

在这个案例中，高达 **45%** 的系统崩溃是由**“搜索引擎返回结果质量低”**引起的。
这就明确地告诉你：**立刻停止去调教什么生成文章的 Prompt！把你团队这周所有的精力，全部投入到优化搜索结果上去！**（比如：更换搜索 API、限制搜索域名只能是 `.edu` 或 `.org`，或者在搜索后增加一个过滤机制）。

当然，在做决策时，还要结合**可行性（Feasibility）**。如果某个组件报错率高达 60%，但你目前受限于技术瓶颈毫无优化思路；而另一个组件报错率 20%，但你只要加一行代码就能解决，那你显然应该先拿那个低垂的果实（Low-hanging fruit）。

**4. Error Analysis 的三步闭环**

作为一名高阶开发者，请将以下三点内化为你的工程本能：

1.  **养成查看 Traces 的习惯**：不要只盯着最后的输出看。只有经常翻阅系统的中间状态，你才能真正培养出对底层逻辑运转的“直觉”。
2.  **进行严谨的错误归因分析**：当最终结果很糟糕时，不要凭空猜测，顺着 Trace 一步步往上游查，找出**第一个**表现明显低于人类水准的组件。
3.  **用数据指引精力的分配**：把团队宝贵的研发时间，精准地投入到那些“错误占比最高且有优化思路”的瓶颈组件上。

错误分析，是系统迭代中最耗时、却也是回报率最高的工作。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415124913522.png)

为了让你对这套方法论有更深刻的体感，在接下来的视频中，我们将跳出研究型 Agent 的框架，去看更多不同业务场景下的错误分析实战案例。准备好迎接更复杂的系统除错挑战了吗？我们下一节见！

### 更多的错误分析

在前面的内容中，我们讨论了如何通过追踪 (Traces) 和错误分析 (Error Analysis) 来打破盲目调优的困境。但正如很多经验丰富的开发者所言：**仅仅知道理论是不够的，你必须亲自查看并分析大量的真实案例，才能真正磨练出对 Agent 系统的“诊断直觉”。**

为了进一步深化这种直觉，让我们深入剖析两个在企业中极其常见的 Agentic 落地场景：发票处理流与客服邮件响应流。看看在这些看似简单的流程中，魔鬼是如何隐藏在细节里的。

**1. 实战案例一：自动化发票处理 (Invoice Processing)**

回顾一下发票处理的标准流水线：
输入 PDF 发票 -> `PDF-to-text API` (转文本) -> `LLM` (提取包含“到期日”在内的 4 个字段) -> `Update Database API` (入库)。

假设在上一节的端到端评估（End-to-End Eval）中，我们发现系统经常提取出**错误的到期日（Due Date）**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415163357547.png)

现在，我们进入**错误分析**环节，将目光锁定在流程中的前两个关键组件上。我们需要找出导致这个错误的根本原因：
*   **嫌疑犯 A**：`PDF-to-text API`。是不是图片转文字的时候乱码了？导致大模型根本看不到正确的日期？
*   **嫌疑犯 B**：`LLM`。是不是文本提取完全正确，但 LLM 的逻辑推理出错了，错误地把“发票开具日”当成了“付款到期日”？

**执行诊断流程：**

1.  **收集“错题本”**：过滤掉那些提取正确的发票，专门挑出 10 到 100 张提取错日期的发票（Focus on examples where performance is subpar）。
2.  **逐案审查 (Case-by-case review)**：建立一个追踪矩阵（Spreadsheet），人工检查每一个失败案例的中间输出（Trace）。

**统计结果可能如下：**

| 失败样本编号     | `PDF-to-text` 组件表现       | `LLM` 信息提取组件表现      |
| :--------------- | :--------------------------- | :-------------------------- |
| **Invoice 1**    | ❌ 提取乱码，文字挤在一起     | (无有效输入，无法评估)      |
| **Invoice 2**    | (文本提取清晰，包含所有日期) | ❌ 错误地选择了 Invoice Date |
| **Invoice 3**    | (文本提取清晰，包含所有日期) | ❌ 错误地选择了 Invoice Date |
| ...              | ...                          | ...                         |
| **总体错误占比** | **15%**                      | **87%**                     |

*(注：百分比相加可能超过 100%，因为同一个案例中可能多个组件同时发生错误，它们并非互斥的。)*

**架构师的决策：**
数据摆在面前，结论显而易见：高达 87% 的错误是由于大模型没有分清两个日期的业务含义导致的。
如果没有做这步错误分析，你的团队可能还在瞎折腾——以为是 OCR（光学字符识别）识别率不够，花几个月时间去微调或更换更贵的 `PDF-to-text` 引擎，结果到头来对最终的业务指标（系统整体表现）毫无帮助。现在你知道了：**只需要去优化大模型提取字段时的 Prompt，加上“务必区分开具日与到期日”的强力约束即可。**

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415163636611.png)

**2. 实战案例二：客服邮件智能响应 (Customer Email Response)**

再来看一个更复杂的工作流。当系统收到一封抱怨“发错货”的客服邮件时：
输入邮件 -> `LLM` (解析邮件并生成数据库查询 SQL) -> `Orders Database API` (执行查询) -> `LLM` (结合订单数据起草回复) -> 推送人工审核。

当人工审核员频繁打回（Reject）这些草稿，认为最终输出“令人不满意”时，我们同样需要顺藤摸瓜。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415163901667.png)

可能出问题的环节有三个：
1.  **LLM drafted query**：生成的 SQL 语法错误，或者查询条件写错。
2.  **Orders database**：数据库本身数据损坏或不一致（比如数据库里记录的也是发错了的货）。
3.  **LLM drafted email**：获取了正确的数据，但生成的回复语气不对，或者没有解决客户的问题。

**执行诊断并统计：**

| 失败样本编号     | LLM 生成 SQL (Database Query) | 数据库系统 (Database Info) | LLM 起草回复 (Draft Email)     |
| :--------------- | :---------------------------- | :------------------------- | :----------------------------- |
| **Email 1**      | ❌ 查错了表 (Wrong table)      | (无正确输入)               | (无正确输入)                   |
| **Email 2**      | (SQL 正确)                    | ❌ 数据库记录丢失           | ❌ 针对丢失记录回复了防御性语气 |
| ...              | ...                           | ...                        | ...                            |
| **总体错误占比** | **75%**                       | **4%**                     | **30%**                        |

**架构师的决策：**
这份追踪表清晰地勾勒出了系统的“疾病图谱”：

*   **首要瓶颈 (75% Error)**：大模型写 SQL 的能力不行，或者它对数据库表结构的理解有偏差。**这是最高优先级的攻坚方向！**
*   **次要瓶颈 (30% Error)**：回复语气不够安抚（如 Defensive tone）。可以在解决 SQL 问题后，通过修改 Prompt 来优化。
*   **非核心问题 (4% Error)**：底层数据库本身的数据脏乱差。这属于业务遗留问题，对 Agent 团队来说投入产出比极低，可以暂缓处理。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415223659984.png)

**3. 总结与预告：从“端到端”走向“组件级”**

在 Agentic AI 的开发中，**错误分析（Error Analysis）是你决定“下一步该去写什么代码”的唯一可靠指南针。** 在任何一个复杂的系统中，总有成百上千个点可以去优化，只有依靠数据驱动，你才能避开“无用功的陷阱”，实现效率的最大化。

现在，你通过错误分析精准定位到了那个拖后腿的组件（比如上例中负责“写 SQL 查询”的那个 LLM）。接下来你要怎么做？

如果每次你修改了写 SQL 的 Prompt，都要把整个系统从头到尾（接收邮件 -> 写SQL -> 查库 -> 写回复）重新跑一遍来验证，那效率实在是太低了。

为了解决这个问题，我们需要引入一种与“端到端评估（End-to-End Evals）”相辅相成的高级评估技术——**组件级评估 (Component-level Evals)**。在下一节中，我们将详细探讨如何对单一组件进行隔离测试，让你的调优效率再上一个台阶！

### 组件级评估

在上一节的错误分析（Error Analysis）中，我们成功定位了拖垮系统整体表现的“罪魁祸首”。

假设在深度研究智能体（Research Agent）的案例中，我们发现 Agent 最终生成的文章总是遗漏核心论点（Missed key points），而根本原因在于工作流前期的 **“Web Search（网络搜索）”** 环节返回了大量低质量的博客，没有抓取到权威信息源。

既然锁定了目标，下一步就是优化搜索引擎的调用策略。但是，我们该如何验证我们的优化是否有效呢？

**1. 端到端评估的困境：昂贵且充满噪音**

初学者通常的做法是：每修改一次搜索策略，就把整个端到端的系统重新跑一遍（生成搜索词 -> 调用搜索 -> LLM 筛选 -> LLM 写草稿）。

这种做法存在两个致命的缺陷：
1.  **极其昂贵（Expensive）**：一次完整的端到端运行不仅消耗大量的 Token 成本，还会耗费漫长的等待时间，这严重拖慢了工程师的迭代节奏。
2.  **噪音干扰（Noise）**：大模型的输出本质上是具有随机性（Temperature）的。即使你偷偷优化了搜索结果，如果下游负责“写草稿”的 LLM 恰好在某次运行中发挥失常，端到端的总分依然可能下降。这种随机性带来的“噪音”，会完全掩盖掉你在单一组件上取得的微小进步。

为了打破这种低效的闭环，我们需要引入高级架构师的必备利器——**组件级评估（Component-level Evaluations）**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415225104556.png)

**2. 实战：如何只评估“Web Search”工具？**

顾名思义，组件级评估就是把出问题的组件从整个工作流中单独“抠”出来，为其量身定制一套专属的微型测试集。

针对 Web Search 组件，我们可以这样构建：

1.  **建立黄金标准池（Gold Standard List）**：
    找人类专家针对几组特定的测试查询（如“黑洞最新科学进展”），精挑细选出最具权威性的 3 到 5 个网页 URL（如 NASA 官网、Nature 论文链接）。这就是我们期望搜索组件必须找到的“黄金资源”。
2.  **编写自动化代码计算重合度**：
    编写一段 Python 脚本，独立执行搜索工具。获取返回的 URL 列表后，将其与我们的“黄金标准池”进行数学比对。
    在信息检索领域，我们可以使用标准的 **F1-score**（综合衡量精确率 Precision 和召回率 Recall 的指标）来量化搜索结果的重合度。
3.  **高频、快速地调优超参数（Hyperparameters）**：
    有了这个只需要几秒钟就能跑完、且结果绝对确定的微型 Eval，你就可以放开手脚去疯狂调参了：
    *   **更换底层引擎**：今天测 Google，明天换 Bing、DuckDuckGo 或是专为大模型设计的 Tavily。
    *   **调整搜索颗粒度**：修改返回结果的数量限制（Number of results）、限定搜索的日期范围（Dates）或强制指定域名（如仅限 `.edu`）。

每一次微小的参数调整，Eval 脚本都会立刻给你一个冷酷而精准的 F1-score 变化趋势，让你明确知道自己是否走在正确的道路上。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415225141144.png)

**3. 组件级评估的三大核心优势**

将复杂系统“化整为零”进行评估，具有不可替代的工程价值：

1.  **提供极其纯粹的信号（Clearer signal）**：
    它彻底屏蔽了工作流中其他大模型节点带来的随机性噪音。你优化的是搜索，你看到的就纯粹是搜索准确率的上升。
2.  **团队协作的效率倍增器（More efficient for focused teams）**：
    在大型企业级 Agent 项目中，通常会有多支团队并行开发。负责检索（Retrieval）的团队只需要死磕他们组件的 F1-score，负责最终写作（Generation）的团队只需要死磕文章维度的 Rubric 评分。各个团队可以在自己的小圈子里高速迭代，互不干扰（Work on smaller, more targeted problems faster）。
3.  **节省巨量成本**：
    在彻底调好一个组件之前，无需浪费 Token 去跑后续的复杂推理步骤。只有当你确信该组件的指标已经达到令人满意的阈值时，再跑一次端到端（End-to-end）测试作为最终的集成验收即可。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260415225249692.png)

**4. 总结与展望**

通过**端到端评估（纵观全局） -> 错误分析（定位病灶） -> 组件级评估（隔离调优）**这套三步走的组合拳，你已经掌握了诊断和重构任何复杂 Agent 系统的系统化方法论。

现在，你已经明确知道需要优化某个特定的组件，并为其搭建好了组件级的测试仪（Eval）。那么，**具体到代码和算法层面，我们究竟有哪些手段可以让一个表现糟糕的组件变得更好？**

在下一节中，我们将卷起袖子，深入探讨优化具体组件的硬核技术手段。敬请期待！

### 优化瓶颈组件

在经历了一系列的追踪 (Traces)、错误分析 (Error Analysis) 和组件级评估 (Component-level Evals) 之后，你终于精准地将系统整体表现不佳的责任，定位到了某一个具体的组件上。

接下来就是见真章的时刻：**我们该如何下手去优化它？**

由于 Agentic 工作流通常由多种异构组件拼接而成，我们必须根据组件的性质（是否基于 LLM），采用截然不同的调优策略。

**1. 策略一：非 LLM 组件 (Non-LLM Components) 的优化**

并非所有的问题都是大模型的锅。你的工作流中可能包含 Web 搜索引擎、RAG（检索增强生成）系统的向量数据库、代码执行沙盒，甚至是一些传统的机器学习模型（如语音识别或物体检测模型）。

对于这类高度工程化或具有确定性的组件，优化手段主要有两种：

**(1) 调参 (Tune Hyperparameters)**
每一个非 LLM 组件都有其关键的控制旋钮（Hyperparameters）。

*   **Web 搜索引擎**：你可以调整返回结果的数量（Number of results）、限制抓取的日期范围（Date range），或者限定搜索特定的域名（如 `.edu`）。
*   **RAG 文本检索**：如果你发现检索出的文档关联性很低，你可以尝试提高“向量相似度阈值（Similarity threshold）”；如果发现大模型总是断章取义，你可能需要调整文档切片的“块大小（Chunk size）”。
*   **传统 ML 模型**：比如在图像人员检测中，你可以调整“检测阈值（Detection threshold）”来平衡假阳性（误报）和假阴性（漏报）。

**(2) 果断替换 (Replace the component)**
在工程实践中，我们经常发现某些开源库或云服务的 API 就是达不到业务要求。此时，最有效的办法就是“换个轮子”。比如，把 Google Search 换成 Bing 或者 Tavily；把底层的向量数据库提供商换掉。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260416090736056.png)

**2. 策略二：LLM 组件 (LLM Components) 的优化**

如果瓶颈确实出在大模型身上（比如它提取错了数据、写错了代码或没有遵循指令），那么我们有四级渐进式的优化武器库。

**(1) 第一级：提示词工程 (Improve your prompts)**
这是成本最低、见效最快的方法。

*   **增加显式指令**：不要含糊其辞。明确告诉模型你想要的输出格式、语气以及必须遵守的限制。
*   **引入 Few-shot prompting（少样本提示）**：在 Prompt 中塞入几个优质的“输入-输出”对照样例。大模型是极度依赖上下文模式匹配的机器，给它看优秀的例子，比给它写长篇大论的指令要有效得多。

**(2) 第二级：模型路由 (Try a new model)**
如果 Prompt 怎么改都没用，可能是当前模型的能力已经见顶了。今天的大模型生态极其繁荣，不同厂商的模型各有所长：有的擅长写代码，有的擅长遵循复杂格式指令，有的则在特定垂直领域的知识储备更深厚。
使用像 `aisuite` 这样的多模型适配库，你可以只需修改一行代码，就能在 OpenAI、Anthropic、Mistral 之间无缝切换，并用你之前写好的 Evals 来测出哪个模型得分最高。

**(3) 第三级：任务拆解 (Split up the step)**
这是一种架构级的降维打击。如果你在一个 Prompt 里让大模型“既要...又要...还要...”，它很容易在处理多重逻辑时发生崩溃或遗漏。
**解法：将一个巨无霸的 LLM 节点，拆解为多个微小、专注的 LLM 节点。**
例如：与其让模型“直接写出一篇没有错别字且符合营销语气的文章”，不如拆解为：LLM 1 负责写初稿 -> LLM 2 负责检查错别字 -> LLM 3 负责调整营销语气。这其实就是我们之前探讨过的“反思模式（Reflection）”的底层逻辑。

**(4) 第四级：模型微调 (Fine-tune a model)**
这是最后的杀手锏。微调需要准备高质量的内部数据集，并且耗费昂贵的算力和大量开发者时间。
通常，**只有当你尝试了前面所有方法，系统性能卡在 90% 或 95%，且你必须榨干最后那几个百分点的准确率时，才会考虑微调。** 对于大多数处于探索期的项目，强烈建议推迟这一步。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260416090929298.png)

**3. 培养“模型直觉 (Model Intuition)”**

在决定是“改 Prompt”还是“换模型”时，资深架构师往往比新手更高效。这是因为他们在大脑中建立了一种对“模型智力（Model Intelligence）”的深刻直觉。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260416091142823.png)

他们清楚地知道：**小模型（如 Llama 3.1 8B）**在回答简单事实时又快又好，但在面对诸如“提取 PII 隐私信息并按特定 JSON 格式输出”这种复杂的“指令遵循（Instruction Following）”任务时，小模型很容易崩溃（输出格式混乱或遗漏关键信息）。此时，他们会果断切换到**前沿大模型（如 GPT-4o 或 Claude 3.5 Sonnet）**。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260416091401289.png)

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260416091908953.png)

如何培养这种极其宝贵的“模型直觉”？吴恩达教授给出了四个绝佳的建议：
1.  **高频试玩新模型**：每次有新模型（无论是闭源还是开源）发布，都用你准备好的一套私人测试集（Personal set of evals）去跑一跑，感受它的能力边界。
2.  **“偷窥”高手的 Prompt**：阅读开源社区（尤其是顶级开源项目）源码里硬编码的 Prompt。看看那些顶尖工程师是如何对大模型发号施令的，这是提升提示词功力最快的捷径。
3.  **多进行模型切换对比**：在你的 Agent 工作流中频繁切换不同模型，观察 Trace 里的中间输出差异。
4.  **建立成本与速度的全局观**：不仅仅评估聪明程度，还要在心里建立起一张不同模型“性能-价格-延迟”的平衡表。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260416092017470.png)

**4. 总结与预告**

通过上述策略，我们终于有能力将 Agentic 系统的质量（Quality）推向极致。

然而，在商业落地中，质量并不是唯一的指标。当你的系统开始服务成千上万的用户时，另外两座大山会立刻横亘在你面前：**极高的延迟（Latency）**和**令人咋舌的 API 账单（Cost）**。

在下一个、也是本模块的最后一个视频中，我们将探讨在 Agentic 工作流中，如何优雅地进行成本控制和速度优化。敬请期待！
