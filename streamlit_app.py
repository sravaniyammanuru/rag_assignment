import streamlit as st

from src.graph import rag_graph


st.set_page_config(page_title="Agentic AI eBook Chat", layout="wide")
st.title("Agentic AI eBook chatbot")
st.caption("Answers are grounded in the PDF. Off-topic questions should get a refusal.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask something from the eBook")

if query:
    result = rag_graph.invoke(
        {
            "question": query,
            "context": [],
            "documents": [],
            "scores": [],
            "answer": "",
            "score": 0.0,
        }
    )
    st.session_state.history.append((query, result))

for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.caption(f"confidence: {result['score']:.4f}")

if st.session_state.history:
    _, latest = st.session_state.history[-1]
    with st.sidebar:
        st.subheader("Retrieved chunks")
        st.write(f"Confidence score: **{latest['score']:.4f}**")
        scores = latest.get("scores") or []
        for i, text in enumerate(latest.get("context") or []):
            label = f"chunk {i + 1}"
            if i < len(scores):
                label += f" (sim={scores[i]:.3f})"
            with st.expander(label):
                st.write(text)
