#!/usr/bin/env python3
"""
All-in-one Streamlit chat UI for MedIntel-AI.
Supports multiple saved chat sessions, New Chat, Clear Chat, and multi-turn memory.
"""
import os
import uuid
import streamlit as st

from src.rag.unified_medical_rag import UnifiedMedicalRAG

st.set_page_config(page_title="MedIntel-AI", page_icon="🩺", layout="wide")


@st.cache_resource
def load_rag() -> UnifiedMedicalRAG:
    return UnifiedMedicalRAG(
        llm_model=os.getenv("MEDINTEL_LLM_MODEL", "qwen2.5:3b")
    )


def init_state() -> None:
    if "chats" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state["chats"] = {
            first_id: {"title": "New Chat", "messages": []}
        }
        st.session_state["current_chat_id"] = first_id


def new_chat() -> None:
    current = st.session_state["chats"][st.session_state["current_chat_id"]]
    # Only create a new session if the current one already has messages
    if not current["messages"]:
        return
    new_id = str(uuid.uuid4())
    st.session_state["chats"][new_id] = {"title": "New Chat", "messages": []}
    st.session_state["current_chat_id"] = new_id


def clear_current_chat() -> None:
    cid = st.session_state["current_chat_id"]
    st.session_state["chats"][cid] = {"title": "New Chat", "messages": []}


def select_chat(chat_id: str) -> None:
    st.session_state["current_chat_id"] = chat_id


rag = load_rag()
init_state()

current_id = st.session_state["current_chat_id"]
current_chat = st.session_state["chats"][current_id]

with st.sidebar:
    st.title("MedIntel-AI")
    col1, col2 = st.columns(2)
    with col1:
        st.button("+ New Chat", on_click=new_chat, type="primary", use_container_width=True)
    with col2:
        st.button("Clear Chat", on_click=clear_current_chat, use_container_width=True)

    st.divider()
    st.subheader("Saved Chats")
    # Show most recent chats at the top
    for cid, chat_data in reversed(list(st.session_state["chats"].items())):
        is_active = cid == current_id
        label = f"▶ {chat_data['title']}" if is_active else chat_data["title"]
        st.button(
            label,
            key=f"chat_{cid}",
            on_click=select_chat,
            args=(cid,),
            use_container_width=True,
        )

    st.divider()
    drugs = ", ".join(d.title() for d in rag.available_drugs)
    st.caption(f"Model: `{rag.llm_model}`")
    st.caption(f"Knowledge: {len(rag.records)} DailyMed chunks ({drugs})")

st.title("MedIntel-AI")
st.caption("Educational Medical Evidence Assistant grounded in FDA DailyMed labels.")

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Cited DailyMed Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], start=1):
                    meta = src["metadata"]
                    st.markdown(
                        f"**[{i}] {meta.get('drug_name', 'N/A').title()} — "
                        f"{meta.get('section', 'N/A')}** *(score: {src['score']:.3f})*"
                    )
                    st.write(src["text"])
                    if meta.get("source_url"):
                        st.markdown(f"[View DailyMed Label]({meta['source_url']})")

if prompt := st.chat_input("Ask about metformin, glimepiride, warnings, dosage..."):
    history_snapshot = list(current_chat["messages"])

    if not current_chat["messages"]:
        current_chat["title"] = prompt[:32] + ("..." if len(prompt) > 32 else "")

    current_chat["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.answer(query=prompt, history=history_snapshot)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"Cited DailyMed Sources ({len(result['sources'])})"):
                for i, src in enumerate(result["sources"], start=1):
                    meta = src["metadata"]
                    st.markdown(
                        f"**[{i}] {meta.get('drug_name', 'N/A').title()} — "
                        f"{meta.get('section', 'N/A')}** *(score: {src['score']:.3f})*"
                    )
                    st.write(src["text"])
                    if meta.get("source_url"):
                        st.markdown(f"[View DailyMed Label]({meta['source_url']})")

    current_chat["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.rerun()
