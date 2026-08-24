# evaluate.py
# Dependencies: same as run_single.py

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from graph import build_graph
from retriever import ExampleRetriever, gold_titles
from run_single import build_llm
from state import initial_state

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def answer_matches(predicted: str, gold: str) -> bool:
    # Simple normalized exact match — good enough for a first pass.
    # Swap in a real F1/EM implementation later for closer parity with
    # HotpotQA's official evaluation script.
    return predicted.strip().lower() == gold.strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of validation examples to evaluate")
    args = parser.parse_args()

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    llm = build_llm()

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    answer_correct = 0
    retrieved_gold_count = 0
    cited_gold_count = 0

    for i in range(args.n):
        example = dataset[i]
        retriever = ExampleRetriever(example)
        graph = build_graph(retriever, llm)

        result = graph.invoke(initial_state(example["question"]))
        gold = example["answer"]
        gold_set = gold_titles(example)
        retrieved_titles = {p["title"] for p in result["retrieved_passages"]}
        cited_set = set(result["cited_titles"])

        a_ok = answer_matches(result["answer"], gold)
        # Did retrieval actually surface both gold paragraphs, regardless of
        # what the LLM later claims to have used?
        retrieved_ok = gold_set.issubset(retrieved_titles)
        # Did the LLM's self-reported citation match — a proxy for grounding,
        # separate from whether retrieval itself succeeded.
        cited_ok = gold_set.issubset(cited_set)

        answer_correct += a_ok
        retrieved_gold_count += retrieved_ok
        cited_gold_count += cited_ok

        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "predicted_answer": result["answer"],
                "gold_answer": gold,
                "answer_correct": a_ok,
                "retrieved_gold": retrieved_ok,
                "cited_gold": cited_ok,
                "cited_titles": "; ".join(result["cited_titles"]),
                "gold_titles": "; ".join(gold_set),
                "hops": result["hop_count"],
            }
        )

        print(f"[{i + 1}/{args.n}] answer={'OK' if a_ok else 'MISS'} retrieved={'OK' if retrieved_ok else 'MISS'} cited={'OK' if cited_ok else 'MISS'}")

    out_path = RESULTS_DIR / "eval_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAnswer accuracy: {answer_correct}/{args.n} ({answer_correct / args.n:.1%})")
    print(f"Retrieval accuracy (gold surfaced): {retrieved_gold_count}/{args.n} ({retrieved_gold_count / args.n:.1%})")
    print(f"Citation accuracy (gold cited): {cited_gold_count}/{args.n} ({cited_gold_count / args.n:.1%})")
    print(f"Per-example results written to {out_path}")


if __name__ == "__main__":
    main()