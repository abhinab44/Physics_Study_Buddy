# LangGraph StateGraph Assembly
# Domain: Study Buddy — Physics


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from study_buddy.state import CapstoneState
from study_buddy.nodes import (
    memory_node,
    router_node,
    retrieval_node,
    skip_retrieval_node,
    tool_node,
    answer_node,
    eval_node,
    save_node,
    FAITHFULNESS_THRESHOLD,
    MAX_EVAL_RETRIES,
)


def route_decision(state: CapstoneState) -> str:
    """After router_node: decide which path to take."""
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    """After eval_node: retry answer or save and finish."""
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


def build_graph():
    """Builds and compiles the StateGraph."""
    graph = StateGraph(CapstoneState)

    # 1. Add all nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    # 2. Add edges
    # Entry point
    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    # Router logic
    graph.add_conditional_edges(
        "router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )

    # Paths converge at answer
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")

    # Eval loop
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    graph.add_edge("save", END)

    # 3. Compile with persistent in-memory checkpointer for thread_id
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    return app


app = build_graph()
