# state.py
# No dependencies — pure stdlib.

from typing import List, Optional, TypedDict


class AgentState(TypedDict):
    question: str  # the original question, never changes
    current_query: str  # the query used for the most recent/next retrieval
    retrieved_passages: List[dict]  # accumulated across hops: [{"title": str, "text": str}, ...]
    hop_count: int  # how many retrieval hops have happened so far
    evidence_sufficient: bool  # set by the assess step; drives the loop-or-continue edge
    answer: Optional[str]  # filled in by the synthesize step
    cited_titles: List[str]  # which paragraph titles the answer actually drew on


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "current_query": question,
        "retrieved_passages": [],
        "hop_count": 0,
        "evidence_sufficient": False,
        "answer": None,
        "cited_titles": [],
    }