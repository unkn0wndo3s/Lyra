"""Download and preprocess open conversational corpora for LLM training."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, List

from datasets import load_dataset

OUTPUT_DIR = Path("data/corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")


def process_ultrachat(limit: int = 100000) -> None:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{limit}]")
    rows: List[dict] = []
    for example in ds:
        messages = example["messages"]
        utts = []
        for msg in messages:
            role = msg["role"]
            text = normalize_text(msg["content"])
            if text:
                utts.append(f"<{role}> {text}")
        if len(utts) < 2:
            continue
        rows.append(
            {
                "source": "ultrachat_200k",
                "language": "en",
                "style": "dialogue",
                "utterances": utts,
            }
        )
    write_jsonl(OUTPUT_DIR / "ultrachat_sample.jsonl", rows)


def process_opus_books(config: str = "en-es", limit: int = 100000) -> None:
    lang_src, lang_tgt = config.split("-")
    ds = load_dataset("Helsinki-NLP/opus_books", config, split=f"train[:{limit}]")
    rows: List[dict] = []
    for example in ds:
        translation = example["translation"]
        src = normalize_text(translation[lang_src])
        tgt = normalize_text(translation[lang_tgt])
        if not src or not tgt:
            continue
        rows.append(
            {
                "source": "opus_books",
                "language_pair": config,
                "style": "sentence_pair",
                "text_src": src,
                "text_tgt": tgt,
            }
        )
    write_jsonl(OUTPUT_DIR / f"opus_books_{config}.jsonl", rows)


def main() -> None:
    print("Downloading UltraChat sample...")
    process_ultrachat()
    print("Downloading OPUS Books sample...")
    process_opus_books()
    print("All corpora downloaded to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()


