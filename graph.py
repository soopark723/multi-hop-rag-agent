# graph.py
# Dependencies:
#   pip install langgraph langchain_core

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from state import AgentState

MAX_HOPS = 3  # hard cap so a stubborn "insufficient" verdict can't loop forever


def build_graph(retriever, llm):
    """Builds a fresh graph closed over one example's retriever and a shared llm.

    Called once per example (retriever is example-specific), reusing the
    same llm instance across calls to avoid re-creating the client each time.
    """

    def retrieve_node(state: AgentState) -> dict:
        results = retriever.search(state["current_query"], top_k=3)
        seen_titles = {p["title"] for p in state["retrieved_passages"]}
        new_passages = [p for p in results if p["title"] not in seen_titles]
        return {
            "retrieved_passages": state["retrieved_passages"] + new_passages,
            "hop_count": state["hop_count"] + 1,
        }

    def assess_node(state: AgentState) -> dict:
        passages_text = "\n\n".join(f"[{p['title']}] {p['text']}" for p in state["retrieved_passages"])
        prompt = f"""You are checking whether enough evidence has been gathered to answer a question.

Question: {state['question']}

Passages gathered so far:
{passages_text}

If these passages together contain enough information to fully answer the question, respond with exactly:
SUFFICIENT: <one sentence explaining why>

If not, respond with exactly:
INSUFFICIENT: <a new, different search query that would help find the missing piece>
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        if text.upper().startswith("SUFFICIENT"):
            return {"evidence_sufficient": True}

        next_query = text.split(":", 1)[1].strip() if ":" in text else state["question"]
        return {"evidence_sufficient": False, "current_query": next_query}

    def synthesize_node(state: AgentState) -> dict:
        passages_text = "\n\n".join(f"[{p['title']}] {p['text']}" for p in state["retrieved_passages"])
        prompt = f"""Answer the question using only the passages below. Be concise — a short phrase is ideal.

Question: {state['question']}

Passages:
{passages_text}

Respond in exactly this format:
ANSWER: <your answer>
SOURCES: <comma-separated list of the paragraph titles you actually used>
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        answer = ""
        sources = []
        for line in text.splitlines():
            if line.upper().startswith("ANSWER:"):
                answer = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SOURCES:"):
                sources = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]

        return {"answer": answer or text, "cited_titles": sources}

    def should_continue(state: AgentState) -> str:
        if state["evidence_sufficient"] or state["hop_count"] >= MAX_HOPS:
            return "synthesize"
        return "retrieve"

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess", assess_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges("assess", should_continue, {"retrieve": "retrieve", "synthesize": "synthesize"})
    builder.add_edge("synthesize", END)

    return builder.compile()