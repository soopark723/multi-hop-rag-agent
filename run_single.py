# run_single.py
# Dependencies:
#   pip install langgraph langchain_huggingface langchain_core datasets rank_bm25 python-dotenv
#
# Requires HF_TOKEN set in your .env — reuses the same Hugging Face route as
# your other repos, not local Ollama, so evaluation stays reasonably fast.

import argparse
import os

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from graph import build_graph
from retriever import ExampleRetriever, gold_titles
from state import initial_state

load_dotenv()


def build_llm():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        task="text-generation",
        provider="nscale",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        max_new_tokens=512,
        temperature=0,
    )
    return ChatHuggingFace(llm=llm_endpoint)


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