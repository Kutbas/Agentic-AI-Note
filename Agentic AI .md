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

