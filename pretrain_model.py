import torch
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
from tokenizers import Tokenizer
from accelerate import Accelerator
import os

# --- CONFIGURATION (Match your VRAM/RAM constraints) ---
OUTPUT_DIR = "./results_pretrain"
TOKENIZER_PATH = "custom_unfiltered_gpt_tokenizer.json"
DATA_PATH = "./data/tokenized_data_grouped" # Assuming the previous script saved the grouped data here

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, unk_token="<|unk|>", pad_token="<|pad|>", trust_remote_code=True)
vocab_size = tokenizer.vocab_size

# 2. Define Custom GPT Architecture (~125M parameters, ideal for 6-8GB VRAM)
#
config = AutoConfig.for_model(
    "gpt2",
    vocab_size=vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,       # Embedding dimension
    n_layer=12,       # Number of transformer layers
    n_head=12,        # Number of attention heads
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

model = AutoModelForCausalLM.from_config(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params:,} parameters.")

# 3. Load Data (Assuming you run the data_setup.py and manually grouped it)
# For this script to work, you would need to group the texts from
# `data_setup.py` and save them locally. Using a placeholder here:
# tokenized_datasets = load_from_disk(DATA_PATH)
# To run this placeholder, please use a small dataset loaded via load_dataset:
tokenized_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=1024, padding="max_length"),
    batched=True, remove_columns=["text"]
)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.05)


# 4. Training Arguments (VRAM-Efficient Settings)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # CRITICAL: Max VRAM usage
    gradient_accumulation_steps=32, # Simulate a batch size of 32
    fp16=True,                      # Use 16-bit precision for memory saving
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    deepspeed="./ds_config_zero1.json", # Link to DeepSpeed config (see Step 3.2)
    gradient_checkpointing=True, # Saves VRAM by re-computing gradients
    ddp_find_unused_parameters=False,
    optim="adamw_hf",
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# 6. Start Pre-training
trainer.train()

# 7. Save Final Model
trainer.save_model(f"{OUTPUT_DIR}/final_pretrained_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_pretrained_model")