# CapstoneState TypedDict
# Single source of truth for all data flowing through the LangGraph.
# Domain: Study Buddy — Physics


from typing import TypedDict, List, Optional


class CapstoneState(TypedDict):
    # Input
    question: str

    # Memory
    messages: List[dict] # [{role, content}, ...]  sliding window
    user_name: Optional[str] # Extracted from 'my name is ...'

    # Routing
    route: str

    # Retrieval
    retrieved: str # Formatted context string '[Topic]\n...'
    sources: List[str] # Topic names from retrieved docs

    # Tool
    tool_result: str # datetime string or error string

    # Answer
    answer: str

    # Evaluation
    faithfulness: float
    eval_retries: int
