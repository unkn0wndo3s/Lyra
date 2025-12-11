import os
import glob
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import AutoTokenizer

# --- CONFIGURATION ---
DATA_DIR = "./data"
MODEL_NAME = "custom_unfiltered_gpt"
VOCAB_SIZE = 32000
BLOCK_SIZE = 1024 # Sequence length for training
TOKENIZER_PATH = f"{MODEL_NAME}_tokenizer.json"

os.makedirs(DATA_DIR, exist_ok=True)
print("Setup complete. Starting data download and tokenization...")

# 1. Download Datasets (Switching to monology/pile-uncopyrighted to fix the script error)
try:
    # Use a dataset that loads cleanly (JSON/Parquet builder)
    # We will load a manageable subset for tokenizer training and a larger subset for pre-training.
    print("Loading Base Corpus (monology/pile-uncopyrighted)...")
    pile_uncopyrighted = load_dataset("monology/pile-uncopyrighted", split="train")

    # Take a manageable number of records for the tokenizer corpus
    tokenizer_data = pile_uncopyrighted.select(range(50000))

    # Load Conversational data for fine-tuning
    print("Loading Conversational Data (daily_dialog)...")
    dialog_dataset = load_dataset("daily_dialog", split="train")

    # Save a small local file for tokenizer training
    with open(f"{DATA_DIR}/tokenizer_corpus.txt", "w", encoding="utf-8") as f:
        for item in tokenizer_data:
            # The 'text' field is used for raw tokenizer training
            f.write(item['text'] + "\n")

    # Save the full dialog dataset for later use
    dialog_dataset.save_to_disk(f"{DATA_DIR}/daily_dialog_disk")

except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Please check your internet connection and Hugging Face Hub access.")
    exit()

# 2. Train Custom BPE Tokenizer
def train_tokenizer(files, vocab_size, save_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # IMPORTANT: Added custom tokens for the Monologue and Tool-Use features
        special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>", "<|tool_call|>", "<|monologue|>"]
    )
    tokenizer.train(files=files, trainer=trainer)

    # Post-processor for single sequence and saving
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}. Vocab size: {tokenizer.get_vocab_size()}")

train_tokenizer(
    files=[f"{DATA_DIR}/tokenizer_corpus.txt"],
    vocab_size=VOCAB_SIZE,
    save_path=TOKENIZER_PATH
)

# 3. Process data for Pre-training (The larger Pile Subset)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, unk_token="<|unk|>", pad_token="<|pad|>")

# Re-load the main dataset for mapping (using a larger subset for training)
# Select the first 1,000,000 documents for pre-training (adjust this number based on time/disk space)
raw_train_data = pile_uncopyrighted.select(range(1000000))

def tokenize_function(examples):
    # Process text content for the model
    return tokenizer(examples["text"], truncation=True, max_length=BLOCK_SIZE)

# Map tokenization to the entire dataset
tokenized_dataset = raw_train_data.map(
    tokenize_function,
    remove_columns=raw_train_data.column_names,
    batched=True,
    num_proc=os.cpu_count() # Use multiple cores for speed
)

# Group texts into fixed-length blocks (CRITICAL for GPT pre-training)
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, making sure all chunks are BLOCK_SIZE.
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Apply the grouping function
grouped_tokenized_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count(),
)

# 4. Save the Final Tokenized Dataset for Phase 3
grouped_tokenized_dataset.save_to_disk(f"{DATA_DIR}/tokenized_data_grouped")

print(f"\nData setup and tokenization script finished. Final grouped dataset saved to {DATA_DIR}/tokenized_data_grouped. Data is ready for Phase 3.")