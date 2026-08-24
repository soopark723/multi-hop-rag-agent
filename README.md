# multi-hop-rag-agent

An agentic RAG system built around [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)'s distractor config — a multi-hop question-answering benchmark where each question needs facts combined from two of ten given Wikipedia paragraphs (the rest are distractors). Unlike single-shot RAG, the agent retrieves, judges whether it has enough evidence, and retrieves again with a refined query if not, before synthesizing a final answer and citing its sources.

## How it works

Every question comes with a fixed set of 10 candidate paragraphs — 2 "gold" paragraphs that actually justify the answer, and 8 distractors chosen to look relevant without being useful. Answering usually requires combining facts from *both* gold paragraphs, and the right search for the second one often isn't obvious until the first has already been read. A single retrieval pass can't reliably solve that, so the agent loops instead of answering off one search.

For one question, the flow through `graph.py` looks like this:

1. **Retrieve** (`retrieve_node`) — BM25 searches the question's 10 paragraphs using the current query (initially, the question itself) and returns the top 3 matches. The 10 paragraphs never change across hops — only the query does.
2. **Assess** (`assess_node`) — the LLM reads everything retrieved so far and judges whether it's enough to answer. If not, it writes a new, more specific query based on what it just learned (e.g. a name or title that surfaced in the last hop), and the graph loops back to step 1.
3. This repeats until the LLM judges the evidence sufficient, or `MAX_HOPS` (3) is reached. The cap is a safety net, not a target — most questions resolve in 2 hops, matching HotpotQA's own two-paragraph design.
4. **Synthesize** (`synthesize_node`) — once satisfied, the LLM writes a final answer and lists which paragraph titles it actually used, so the answer can be checked for grounding rather than taken on faith.

`run_single.py` runs this loop once, on one question, and prints every step of it — the fast way to see what the agent is actually doing. `evaluate.py` runs it across many questions with no interaction needed, and since every HotpotQA row already ships with the correct `answer` and `supporting_facts`, it scores two things separately: whether the final answer matched, and whether the paragraphs actually retrieved/cited matched the real evidence. That split matters — it tells you whether a wrong answer came from bad retrieval or from bad reasoning over evidence that was actually found correctly.

## Project layout

| File | What it does |
|---|---|
| `state.py` | The `AgentState` shape shared across every node, plus `initial_state()`. |
| `retriever.py` | `ExampleRetriever`: a BM25 index built fresh per question, scoped to just that question's 10 candidate paragraphs. |
| `graph.py` | The actual LangGraph: retrieve → assess → loop-or-synthesize. |
| `run_single.py` | Runs one validation example through the graph and prints the full trace, the fast dev loop. |
| `evaluate.py` | Runs N validation examples, scores answer accuracy and retrieval accuracy separately, writes results to `results/eval_results.csv`. |

## Setup

```powershell
pip install langgraph langchain_huggingface langchain_core datasets rank_bm25 python-dotenv
```

### Environment variables

```dotenv
HF_TOKEN=your_huggingface_token
```

This uses Hugging Face Inference Providers, same as the LangGraph mail-sorting repo, via `Qwen/Qwen2.5-Coder-32B-Instruct` on the `nscale` provider.

## Running it

```powershell
python run_single.py --index 0
python evaluate.py --n 50
```

## Design notes

- Uses the **distractor** config, not fullwiki. Retrieval is scoped to each question's given 10 paragraphs rather than a full Wikipedia corpus, so there's no persistent index to build.
- No embeddings. Retrieval is plain BM25 (`rank_bm25`), rebuilt fresh per question.
- `evaluate.py` reports two separate accuracy numbers on purpose: whether retrieval actually surfaced the gold paragraphs, and whether the LLM's self-reported citations matched — splitting retrieval failures from grounding/reasoning failures.
- `answer_matches()` is a simple exact-string match, not HotpotQA's official F1/EM metric. Good enough to get moving, worth upgrading later.

## Acknowledgments

The scenario is inspired by Unit 3 ("Agentic RAG") of Hugging Face's [Agents Course](https://huggingface.co/learn/agents-course), adapted for the HotpotQA dataset rather than the course's own guest-list example.
