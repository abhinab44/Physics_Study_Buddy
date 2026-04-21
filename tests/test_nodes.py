# Unit tests for individual isolated node functions.
# Domain: Study Buddy — Physics

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from study_buddy.state import CapstoneState
from study_buddy.nodes import (
    memory_node,
    router_node,
    retrieval_node,
    skip_retrieval_node,
    tool_node,
    answer_node,
    eval_node,
    save_node
)

# Test each node individually


def test_memory_node():
    print("Testing memory_node...")
    test_state = {"question": "What is Newton's law?", "messages": []}
    result = memory_node(test_state)
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == "What is Newton's law?"
    print("PASS memory_node works")


def test_memory_name_extraction():
    print("\nTesting memory_node name extraction...")
    test_state = {
        "question": "My name is Abhinab",
        "messages": []
    }
    result = memory_node(test_state)
    assert result["user_name"] == "Abhinab"
    print("PASS memory_node extracts user name")


def test_router_node():
    print("\nTesting router_node...")
    test_state_memory = {
        "question": "What did you just say?",
        "messages": [{"role": "user", "content": "hi"}]
    }
    result1 = router_node(test_state_memory)
    assert result1["route"] == "memory_only"

    test_state_tool = {
        "question": "What is the current time?",
        "messages": []
    }
    result2 = router_node(test_state_tool)
    assert result2["route"] == "tool"

    test_state_retrieve = {
        "question": "What is Coulomb's law?",
        "messages": []
    }
    result3 = router_node(test_state_retrieve)
    assert result3["route"] == "retrieve"

    print("PASS router_node works (memory, tool, retrieve)")


def test_retrieval_node():
    print("\nTesting retrieval_node...")
    test_state = {"question": "What is Ohm's law?"}
    result = retrieval_node(test_state)
    assert len(result["sources"]) > 0
    assert len(result["retrieved"]) > 50
    print("PASS retrieval_node works")


def test_tool_node():
    print("\nTesting tool_node...")
    test_state = {"question": "time"}
    result = tool_node(test_state)
    assert "Current date" in result["tool_result"]
    print("PASS tool_node works")


def test_save_node():
    print("\nTesting save_node...")
    test_state = {
        "answer": "Force equals mass times acceleration.",
        "messages": [{"role": "user", "content": "hi"}],
        "eval_retries": 1
    }
    result = save_node(test_state)
    assert len(result["messages"]) == 2
    assert result["messages"][-1]["role"] == "assistant"
    assert result["eval_retries"] == 0
    print("PASS save_node works")


if __name__ == "__main__":
    print("=" * 40)
    print("Running Node Unit Tests - Physics")
    print("=" * 40)

    test_memory_node()
    test_memory_name_extraction()
    test_router_node()
    test_retrieval_node()
    test_tool_node()
    test_save_node()

    print("=" * 40)
    print("All isolated node tests passed!")
