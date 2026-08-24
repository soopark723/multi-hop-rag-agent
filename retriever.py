# retriever.py
# Dependencies:
#   pip install datasets rank_bm25
#   C:\Python314\python.exe -m pip install datasets rank_bm25

from typing import Dict, List

from rank_bm25 import BM25Okapi


class ExampleRetriever:
    """A BM25 index scoped to one HotpotQA example's 10 candidate paragraphs.

    HotpotQA's `datasets` loader represents `context` as a dict of parallel
    lists (context["title"][i] / context["sentences"][i]), not a list of
    per-paragraph dicts — see the dataset's own README example. This class
    handles that unpacking once, so nothing else in the project needs to
    know about it.
    """

    def __init__(self, example: dict):
        titles = example["context"]["title"]
        sentence_lists = example["context"]["sentences"]

        self.passages: List[Dict[str, str]] = [
            {"title": title, "text": " ".join(sentences)}
            for title, sentences in zip(titles, sentence_lists)
        ]

        tokenized = [p["text"].lower().split() for p in self.passages]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Return the top_k passages most relevant to `query`."""
        scores = self._bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.passages[i] for i in ranked_indices[:top_k]]


def gold_titles(example: dict) -> set:
    """Ground-truth paragraph titles for an example, per supporting_facts."""
    return set(example["supporting_facts"]["title"])


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    example = dataset[0]

    retriever = ExampleRetriever(example)
    print("Question:", example["question"])
    print("Gold titles:", gold_titles(example))
    print("\nTop 3 passages for the question itself:")
    for passage in retriever.search(example["question"]):
        print(f"- {passage['title']}")