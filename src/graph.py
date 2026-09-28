from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import WEAK_MATCH_THRESHOLD
from src.models import get_llm
from src.retriever import retrieve_documents_with_scores

REFUSAL = "I cannot answer this from the provided eBook."


class AgentState(TypedDict):
    question: str
    context: list[str]
    documents: list
    scores: list[float]
    answer: str
    score: float


def retrieve_node(state: AgentState):
    hits = retrieve_documents_with_scores(state["question"])
    # cover pages / headers embed well but have almost no answer text
    usable = [(doc, score) for doc, score in hits if len((doc.page_content or "").strip()) >= 250]
    if not usable:
        usable = hits
    documents = [document for document, _score in usable]
    scores = [float(_score) for _doc, _score in usable]
    context = [doc.page_content for doc in documents]
    confidence = max(scores) if scores else 0.0

    return {
        "documents": documents,
        "context": context,
        "scores": scores,
        "score": confidence,
    }


def _as_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        bits = []
        for block in content:
            if isinstance(block, str):
                bits.append(block)
            elif isinstance(block, dict):
                bits.append(str(block.get("text") or ""))
        return "\n".join(bits).strip()
    return str(content).strip()


def generate_node(state: AgentState):
    context_bits = state.get("context") or []
    if not context_bits:
        return {"answer": REFUSAL, "score": 0.0}

    context_str = "\n\n---\n\n".join(context_bits)

    prompt = f"""You are answering questions about the Agentic AI eBook.

Use ONLY the context. If the answer is not in the context, reply with exactly:
{REFUSAL}

Do not bring in outside knowledge, even if you know the answer.

Context:
{context_str}

Question:
{state["question"]}
"""

    llm = get_llm()
    response = llm.invoke(prompt)
    answer = _as_text(response.content)

    confidence = float(state.get("score") or 0.0)
    refused = REFUSAL.lower() in answer.lower()
    if refused:
        confidence = min(confidence, 0.12)
    elif confidence < WEAK_MATCH_THRESHOLD:
        # retrieval was basically noise; don't let the model "answer" from it
        answer = REFUSAL
        confidence = min(confidence, 0.12)

    return {"answer": answer, "score": round(confidence, 4)}


def build_rag_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


rag_graph = build_rag_graph()
