# End-to-End graph traversal tests
# Domain: Study Buddy — Physics

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from study_buddy.graph import app


def ask(question, thread_id="t1"):
    print(f"\n--- Q: {question} ---")
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    for chunk in app.stream(
        {"question": question}, config=config,
        stream_mode="updates"
    ):
        for node_name, state_update in chunk.items():
            print(f"[{node_name}] executed")
            if "route" in state_update:
                print(f"  -> route: {state_update['route']}")
            if "faithfulness" in state_update:
                print(
                    f"  -> faithfulness: "
                    f"{state_update['faithfulness']}"
                )
            final_state = state_update

    full_state = app.get_state(config).values
    answer = full_state.get("answer", "")
    print(f"\nA: {answer}\n")
    return full_state


if __name__ == "__main__":
    print("Testing Graph ...")

    # 1. Physics domain retrieval
    ask("What is Newton's second law of motion?", thread_id="t1")

    # 2. Follow-up (memory + retrieval)
    ask("What about the third law?", thread_id="t1")

    # 3. Tool test
    ask("What is the current time?", thread_id="t2")

    # 4. Memory-only test
    ask("What did I just ask you?", thread_id="t2")

    # 5. Name extraction test
    ask("My name is Abhinab", thread_id="t3")
    state = ask("What is my name?", thread_id="t3")
    name = state.get("user_name", "")
    print(f"Extracted name: {name}")

    print("Tests completed.")
