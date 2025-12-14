import os
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import PreTrainedTokenizerFast
from multiprocessing import freeze_support

# ================= CONFIG =================
DATA_DIR = "./data"
MODEL_NAME = "Lyra"

VOCAB_SIZE = 32000
BLOCK_SIZE = 1024

TOKENIZER_JSON = f"{MODEL_NAME}_tokenizer.json"
TOKENIZER_CORPUS = f"{DATA_DIR}/tokenizer_corpus.txt"
TOKENIZED_OUT = f"{DATA_DIR}/tokenized_blocks"

MAX_TOKENIZER_DOCS = 50_000      # pour entraîner le tokenizer
MAX_PRETRAIN_DOCS = 200_000      # réaliste sur 14 Go RAM

os.makedirs(DATA_DIR, exist_ok=True)


def main():
    print("=== DATA SETUP START ===")

    # ================= 1. LOAD DATASET (STREAMING) =================
    print("Loading Pile (streaming mode)...")

    pile = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )

    # ================= 2. BUILD TOKENIZER CORPUS =================
    print("Building tokenizer corpus...")

    with open(TOKENIZER_CORPUS, "w", encoding="utf-8") as f:
        for i, sample in enumerate(pile):
            f.write(sample["text"] + "\n")
            if i + 1 >= MAX_TOKENIZER_DOCS:
                break

    print(f"Tokenizer corpus written ({MAX_TOKENIZER_DOCS} docs)")

    # ================= 3. TRAIN TOKENIZER =================
    print("Training BPE tokenizer...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[
            "<|endoftext|>",
            "<|unk|>",
            "<|pad|>",
            "<|tool_call|>",
            "<|monologue|>",
        ],
    )

    tokenizer.train(files=[TOKENIZER_CORPUS], trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(TOKENIZER_JSON)

    print(f"Tokenizer saved → {TOKENIZER_JSON}")

    # ================= 4. LOAD TOKENIZER (HF FAST) =================
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_JSON,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        eos_token="<|endoftext|>",
    )

    # ================= 5. RELOAD DATASET (STREAMING AGAIN) =================
    print("Reloading Pile for pretraining pass...")

    pile = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )

    def text_generator():
        for i, sample in enumerate(pile):
            if i >= MAX_PRETRAIN_DOCS:
                break
            yield {"text": sample["text"]}

    print("Building pretraining dataset...")
    raw_dataset = Dataset.from_generator(text_generator)

    # ================= 6. TOKENIZE =================
    def tokenize(batch):
        return hf_tokenizer(
            batch["text"],
            truncation=True,
            max_length=BLOCK_SIZE,
        )

    tokenized = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    # ================= 7. GROUP INTO FIXED BLOCKS =================
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

        result = {
            k: [v[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts,
        batched=True,
    )

    # ================= 8. SAVE =================
    grouped.save_to_disk(TOKENIZED_OUT)

    print("\n=== DONE ===")
    print(f"Tokenized dataset saved to: {TOKENIZED_OUT}")


if __name__ == "__main__":
    freeze_support()  # obligatoire sur Windows
    main()
