# run_single.py
# Dependencies:
#   pip install langgraph langchain_ollama langchain_core datasets rank_bm25 python-dotenv

import argparse

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from graph import build_graph
from retriever import ExampleRetriever, gold_titles
from state import initial_state

load_dotenv()


def build_llm():
    return ChatOllama(model="qwen2.5", temperature=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Validation set row to run")
    args = parser.parse_args()

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    example = dataset[args.index]

    retriever = ExampleRetriever(example)
    llm = build_llm()
    graph = build_graph(retriever, llm)

    print(f"Question: {example['question']}")
    print(f"Gold titles: {gold_titles(example)}\n")

    result = graph.invoke(initial_state(example["question"]))

    print(f"Hops taken: {result['hop_count']}")
    print(f"Retrieved: {[p['title'] for p in result['retrieved_passages']]}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Cited: {result['cited_titles']}")
    print(f"Gold answer: {example['answer']}")


if __name__ == "__main__":
    main()