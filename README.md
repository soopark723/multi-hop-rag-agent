# multi-hop-rag-agent

An agentic RAG system built around [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)'s distractor config — a multi-hop question-answering benchmark where each question needs facts combined from two of ten given Wikipedia paragraphs (the rest are distractors). Unlike single-shot RAG, the agent retrieves, judges whether it has enough evidence, and retrieves again with a refined query if not — up to 3 hops — before synthesizing a final answer and citing its sources.

## Project layout

| File | What it does |
|---|---|
| `state.py` | The `AgentState` shape shared across every node, plus `initial_state()`. |
| `retriever.py` | `ExampleRetriever` — a BM25 index built fresh per question, scoped to just that question's 10 candidate paragraphs. |
| `graph.py` | The actual LangGraph: retrieve → assess → loop-or-synthesize. |
| `run_single.py` | Runs one validation example through the graph and prints the full trace — the fast dev loop. |
| `evaluate.py` | Runs N validation examples, scores answer accuracy and retrieval accuracy separately, writes results to `results/eval_results.csv`. |

## Setup

```powershell
pip install langgraph langchain_huggingface langchain_core datasets rank_bm25 python-dotenv
```

### Environment variables

```dotenv
HF_TOKEN=your_huggingface_token
```

No OpenAI key, no local Ollama models — this uses Hugging Face Inference Providers, same as the LangGraph mail-sorting repo, via `Qwen/Qwen2.5-Coder-32B-Instruct` on the `nscale` provider.

## Running it

```powershell
python run_single.py --index 0
python evaluate.py --n 50
```

## Design notes

- Uses the **distractor** config, not fullwiki — retrieval is scoped to each question's given 10 paragraphs rather than a full Wikipedia corpus, so there's no persistent index to build.
- No embeddings — retrieval is plain BM25 (`rank_bm25`), rebuilt fresh per question.
- `evaluate.py` reports two separate accuracy numbers on purpose: whether retrieval actually surfaced the gold paragraphs, and whether the LLM's self-reported citations matched — splitting retrieval failures from grounding/reasoning failures.
- `answer_matches()` is a simple exact-string match, not HotpotQA's official F1/EM metric — good enough to get moving, worth upgrading later.

## Acknowledgments

The scenario is inspired by Unit 3 ("Agentic RAG") of Hugging Face's [Agents Course](https://huggingface.co/learn/agents-course), adapted for the HotpotQA dataset rather than the course's own guest-list example.
