# Streamlit UI for the Physics Study Buddy
# Domain: Study Buddy — Physics


import streamlit as st
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from study_buddy.graph import app as agent_app
from study_buddy.knowledge_base import DOCUMENTS

# App Configuration
st.set_page_config(
    page_title="Physics Study Buddy",
    page_icon="⚛️",
    layout="centered"
)

st.title("⚛️ Physics Study Buddy")
st.caption(
    "Your personalized assistant for B.Tech Engineering Physics."
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This agent uses LangGraph, Groq, and ChromaDB to answer "
        "questions about the B.Tech Engineering Physics syllabus "
        "using retrieval-augmented generation (RAG)."
    )
    st.write(f"Session ID: `{st.session_state.thread_id}`")
    st.divider()

    st.write("**Topics Covered in Knowledge Base:**")
    for d in DOCUMENTS:
        st.write(f"• {d['topic']}")

    st.divider()
    if st.button("📰 New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about Physics..."):
    # Append user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.write(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
            try:
                result = agent_app.invoke(
                    {"question": prompt}, config=config
                )

                answer = result.get(
                    "answer", "Error: No answer generated."
                )
                faithfulness = result.get("faithfulness", None)
                route = result.get("route", None)

                metrics_text = ""
                if route == "retrieve" and faithfulness is not None:
                    metrics_text = (
                        f"\n\n*(Faithfulness: {faithfulness:.2f})*"
                    )

                full_reply = answer + metrics_text
                st.write(full_reply)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_reply}
                )
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
