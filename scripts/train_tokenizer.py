"""Train a custom subword tokenizer on cleaned corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

CORPUS_DIR = Path("data/corpus")
OUTPUT_DIR = Path("data/tokenizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "<BOS>",
    "<EOS>",
    "<user>",
    "<assistant>",
    "<system>",
    "<turn>",
]


def iter_corpus() -> Iterable[str]:
    for path in CORPUS_DIR.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "utterances" in sample:
                    yield " ".join(sample["utterances"])
                elif "text_src" in sample and "text_tgt" in sample:
                    yield f"{sample['text_src']} <turn> {sample['text_tgt']}"
                elif "text" in sample:
                    yield sample["text"]


def train_tokenizer(vocab_size: int = 32000) -> None:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_corpus(), trainer=trainer, length=None)

    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))
    config = {
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "normalization": "NFKC+Lowercase",
        "pre_tokenizer": "Whitespace",
    }
    with (OUTPUT_DIR / "tokenizer_config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    print("Tokenizer saved to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    train_tokenizer()



