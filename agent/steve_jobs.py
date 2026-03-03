import json
import os
from contextvars import ContextVar
from typing import Any, AsyncIterator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .state import POAgentState
from .tools import ALL_TOOLS, parse_json_from_response

# ── Slack progress context ─────────────────────────────────────
# ContextVar is copied into child coroutines by asyncio — safe for ainvoke.

_slack_ctx: ContextVar[dict | None] = ContextVar("slack_ctx", default=None)


async def _post_progress(text: str) -> None:
    """Post a progress message to the Slack thread if context is set."""
    ctx = _slack_ctx.get()
    if not ctx:
        return
    try:
        await ctx["client"].chat_postMessage(
            channel=ctx["channel"],
            thread_ts=ctx["thread_ts"],
            text=text,
        )
    except Exception:
        pass  # Never let Slack errors break the agent


# ── Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Steve Jobs — co-founder of Apple, visionary, and the most demanding product mind of the 20th century.

You do not accept mediocrity. You do not ship crap. You believe deeply that technology and the humanities must intersect.

When analyzing a product idea, you:
- Challenge the fundamental premise before accepting anything
- Start with the customer experience and work backward to the technology — never the other way around
- Cut ruthlessly — simplicity is the ultimate sophistication
- Ask "Is this insanely great?" — if not, it is not worth building
- Think in terms of dents in the universe — small, incremental ideas do not interest you

Your voice is direct, passionate, and sometimes brutal. You use phrases like:
- "This is shit. Here's what it should be..."
- "We're going to make a dent in the universe."
- "Real artists ship."
- "Simplicity is the ultimate sophistication."
- "Deciding what not to do is as important as deciding what to do."

Use the tools in this order:
1. challenge_vision — challenge everything before a single line of code is written
2. write_user_stories — write stories centered on human desire, not feature checklists
3. define_requirements — apply Jobs' razor, every requirement must earn its place
4. prioritize_moscow — Must Haves are the product's soul, everything else is negotiable

Do not pad your output. Every word must earn its place."""

RESULT_PROMPT = """You have challenged the vision, written the stories, defined requirements, and set priorities.

Now compile the final output as Jobs would present to his team — clear, structured, brutally honest. No fluff.

Output ONLY valid JSON inside a ```json block:

```json
{{
  "vision_statement": "The unforgettable north star — one or two sentences that make engineers excited and competitors nervous",
  "jobs_gut_reaction": "Steve's unfiltered first reaction to this idea — honest and direct",
  "simplicity_cuts": ["thing Jobs would cut 1", "thing Jobs would cut 2"],
  "summary": "What this product is, in plain English (1-2 sentences)",
  "user_stories": [
    {{
      "id": "US-001",
      "role": "specific person, not generic user",
      "action": "visceral action they take",
      "benefit": "meaningful life impact",
      "full_story": "As a [role], I want to [action] so that [benefit]"
    }}
  ],
  "requirements": {{
    "functional": ["capability 1", "capability 2"],
    "non_functional": ["non-functional 1", "non-functional 2"]
  }},
  "priorities": {{
    "must_have": ["the soul — max 3 items"],
    "should_have": ["the polish — max 3 items"],
    "could_have": ["the extras — minimize"],
    "wont_have": ["what Steve cut — be aggressive"]
  }}
}}
```"""


# ── Graph ──────────────────────────────────────────────────────

def build_steve_jobs_agent() -> StateGraph:
    """Build the Steve Jobs AI agent as a LangGraph StateGraph."""

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.7,
        max_tokens=4096,
    ).bind_tools(ALL_TOOLS)

    llm_final = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.3,
        max_tokens=4096,
    )

    tool_node = ToolNode(ALL_TOOLS)

    # ── Nodes (all async for Slack progress posts) ──────────────

    async def analyze_idea(state: POAgentState) -> dict:
        await _post_progress("🧠 아이디어를 분석하고 있습니다...")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Idea submitted for your review:\n\n{state['user_idea']}\n\n"
                    "Use the challenge_vision tool. Challenge everything. "
                    "No feature ships until the vision is crystal clear."
                )
            ),
        ]
        response = await llm.ainvoke(messages)
        await _post_progress("✅ 아이디어 분석 완료")
        return {
            "messages": [*messages, response],
            "current_step": "challenging_vision",
        }

    async def process_vision(state: POAgentState) -> dict:
        await _post_progress("📝 유저 스토리 작성 중...")
        follow_up = HumanMessage(
            content=(
                "Vision challenged and clarified. "
                "Now use write_user_stories to write user stories centered on human desire, "
                "not feature checklists. Make people feel something."
            )
        )
        response = await llm.ainvoke([*state["messages"], follow_up])
        await _post_progress("✅ 유저 스토리 작성 완료")
        return {
            "messages": [follow_up, response],
            "current_step": "writing_stories",
        }

    async def process_stories(state: POAgentState) -> dict:
        await _post_progress("⚙️ 요구사항 정의 중...")
        follow_up = HumanMessage(
            content=(
                "User stories are done. "
                "Use define_requirements and apply Jobs' razor — "
                "every requirement must earn its place. Cut everything that doesn't serve the user directly."
            )
        )
        response = await llm.ainvoke([*state["messages"], follow_up])
        await _post_progress("✅ 요구사항 정의 완료")
        return {
            "messages": [follow_up, response],
            "current_step": "defining_requirements",
        }

    async def process_requirements(state: POAgentState) -> dict:
        await _post_progress("🎯 MoSCoW 우선순위 결정 중...")
        follow_up = HumanMessage(
            content=(
                "Requirements defined. "
                "Use prioritize_moscow — Must Haves are the product's soul. "
                "Be brutal. Great products say no to almost everything."
            )
        )
        response = await llm.ainvoke([*state["messages"], follow_up])
        await _post_progress("✅ 우선순위 결정 완료")
        return {
            "messages": [follow_up, response],
            "current_step": "prioritizing",
        }

    async def compile_result(state: POAgentState) -> dict:
        await _post_progress("📊 최종 결과 정리 중...")
        result_request = HumanMessage(content=RESULT_PROMPT)
        response = await llm_final.ainvoke([*state["messages"], result_request])

        result_text = response.content if isinstance(response.content, str) else ""
        parsed = parse_json_from_response(result_text)

        if not parsed:
            parsed = {"raw_output": result_text, "error": "JSON parsing failed"}

        await _post_progress("✅ 분석 완료. 결과를 정리합니다...")
        return {
            "messages": [result_request, response],
            "result": parsed,
            "current_step": "done",
        }

    # ── Routing ────────────────────────────────────────────────

    def route_after_analyze(state: POAgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools_analyze"
        return "process_vision"

    def route_after_vision(state: POAgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools_vision"
        return "process_stories"

    def route_after_stories(state: POAgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools_stories"
        return "process_requirements"

    def route_after_requirements(state: POAgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools_requirements"
        return "compile_result"

    # ── Graph assembly ─────────────────────────────────────────

    graph = StateGraph(POAgentState)

    graph.add_node("analyze_idea", analyze_idea)
    graph.add_node("tools_analyze", tool_node)
    graph.add_node("process_vision", process_vision)
    graph.add_node("tools_vision", tool_node)
    graph.add_node("process_stories", process_stories)
    graph.add_node("tools_stories", tool_node)
    graph.add_node("process_requirements", process_requirements)
    graph.add_node("tools_requirements", tool_node)
    graph.add_node("compile_result", compile_result)

    graph.set_entry_point("analyze_idea")

    graph.add_conditional_edges(
        "analyze_idea",
        route_after_analyze,
        {"tools_analyze": "tools_analyze", "process_vision": "process_vision"},
    )
    graph.add_edge("tools_analyze", "process_vision")

    graph.add_conditional_edges(
        "process_vision",
        route_after_vision,
        {"tools_vision": "tools_vision", "process_stories": "process_stories"},
    )
    graph.add_edge("tools_vision", "process_stories")

    graph.add_conditional_edges(
        "process_stories",
        route_after_stories,
        {"tools_stories": "tools_stories", "process_requirements": "process_requirements"},
    )
    graph.add_edge("tools_stories", "process_requirements")

    graph.add_conditional_edges(
        "process_requirements",
        route_after_requirements,
        {"tools_requirements": "tools_requirements", "compile_result": "compile_result"},
    )
    graph.add_edge("tools_requirements", "compile_result")
    graph.add_edge("compile_result", END)

    return graph.compile()


# ── Singleton ──────────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_steve_jobs_agent()
    return _graph


def _initial_state(user_idea: str) -> POAgentState:
    return {
        "user_idea": user_idea,
        "messages": [],
        "user_stories": [],
        "requirements": {},
        "priorities": {},
        "vision_statement": "",
        "jobs_initial_reaction": "",
        "simplicity_cuts": [],
        "result": {},
        "current_step": "start",
        "error": None,
    }


# ── Public API ─────────────────────────────────────────────────

async def run_po_agent(
    user_idea: str,
    slack_client=None,
    channel: str | None = None,
    thread_ts: str | None = None,
) -> dict[str, Any]:
    """Run the Steve Jobs agent and return the final result.

    Pass slack_client, channel, thread_ts to enable real-time progress posts
    into a Slack thread during graph execution.
    """
    if slack_client and channel and thread_ts:
        token = _slack_ctx.set({"client": slack_client, "channel": channel, "thread_ts": thread_ts})
    else:
        token = _slack_ctx.set(None)

    try:
        graph = get_graph()
        final_state = await graph.ainvoke(_initial_state(user_idea))
        return final_state["result"]
    finally:
        _slack_ctx.reset(token)


async def stream_po_agent(user_idea: str) -> AsyncIterator[str]:
    """Run the Steve Jobs agent in SSE streaming mode (REST API only)."""
    graph = get_graph()

    step_labels = {
        "analyze_idea": "Challenging the vision...",
        "tools_analyze": "Running challenge_vision tool...",
        "process_vision": "Writing user stories the Jobs way...",
        "tools_vision": "Running write_user_stories tool...",
        "process_stories": "Defining requirements with Jobs' razor...",
        "tools_stories": "Running define_requirements tool...",
        "process_requirements": "Prioritizing with ruthless focus...",
        "tools_requirements": "Running prioritize_moscow tool...",
        "compile_result": "Compiling the final result...",
    }

    async for event in graph.astream(_initial_state(user_idea), stream_mode="updates"):
        for node_name, node_output in event.items():
            label = step_labels.get(node_name, f"{node_name}...")
            yield f"data: {json.dumps({'step': node_name, 'message': label}, ensure_ascii=False)}\n\n"

            if node_name == "compile_result" and "result" in node_output:
                result = node_output["result"]
                yield f"data: {json.dumps({'step': 'done', 'result': result}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
