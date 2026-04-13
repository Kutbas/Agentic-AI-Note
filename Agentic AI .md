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

*   **挑战**：大家都知道，LLM 极度不擅长精确的字数控制（要求写 500 字，它经常写 600 字）。
*   **外部反馈工具**：一个简单的 `len(text.split())` 函数。
*   **反思闭环**：如果初稿字数超标，将真实的 `word_count` 数字反馈给 LLM：“*你的文章有 612 个字，超过了 500 字的限制，请大幅删减。*”有了具体的数字反馈，模型就能更精准地压缩篇幅。

![](https://graphbed-1331926955.cos.ap-shanghai.myqcloud.com/GraphBed/20260413140901743.png)

**3. 结语与预告**

反思模式（Reflection），特别是结合了外部反馈的反思模式，是构建高可用 Agent 系统的第一块基石。它告诉我们一个核心的工程哲学：**不要强求大模型一次性完美，而是建立一个能让它不断试错、获取反馈并自我纠正的系统机制。**

我们在这节课中看到了许多“外部工具”的影子（如执行代码、网络搜索、正则匹配）。这正是我们通往更高级 Agentic 架构的桥梁。

在接下来的全新模块中，我们将正式深入探讨 Agentic AI 的第二大核心设计模式——**工具调用 (Tool Use)**。我们将学习如何系统性地赋予大模型调用外部函数的权利，让你的 Agent 彻底长出触角，连接广阔的真实世界！敬请期待！

