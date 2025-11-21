"""Minimal training loop for our custom conversational LLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


class JsonlDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        tokenizer: PreTrainedTokenizerFast,
        context_length: int,
        max_sequences: int | None = None,
    ) -> None:
        self.examples = []
        self.context_length = context_length
        for path in paths:
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    if "utterances" in sample:
                        text = " ".join(sample["utterances"])
                    elif "text_src" in sample and "text_tgt" in sample:
                        text = f"{sample['text_src']} <turn> {sample['text_tgt']}"
                    elif "text" in sample:
                        text = sample["text"]
                    else:
                        continue
                    ids = tokenizer.encode(text)
                    if len(ids) < 4:
                        continue
                    for i in range(0, len(ids), context_length):
                        chunk = ids[i : i + context_length]
                        if len(chunk) < 4:
                            continue
                        self.examples.append(chunk)
                        if max_sequences and len(self.examples) >= max_sequences:
                            return

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def collate(batch, pad_token_id: int):
    max_len = max(item.size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        length = item.size(0)
        input_ids[i, :length] = item
        attention_mask[i, :length] = 1
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, required=True)
    parser.add_argument(
        "--dataset_paths", type=Path, nargs="+", required=True, help="JSONL corpora"
    )
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_sequences", type=int, default=2000)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/run1"))
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer_path))
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "<BOS>"
    tokenizer.eos_token = "<EOS>"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<user>", "<assistant>", "<system>", "<turn>"]}
    )

    dataset = JsonlDataset(
        args.dataset_paths, tokenizer, args.context_length, args.max_sequences
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    with args.config.open() as fp:
        config_dict = json.load(fp)
    config = GPT2Config(**config_dict)
    config.vocab_size = len(tokenizer)

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_interval == 0:
                ppl = math.exp(loss.item())
                print(
                    f"Epoch {epoch} Step {global_step}: loss={loss.item():.4f} ppl={ppl:.2f}"
                )

        ckpt_path = args.output_dir / f"epoch_{epoch}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    print("Training complete. Final checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()



