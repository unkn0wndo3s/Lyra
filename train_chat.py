import os
import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from functools import partial

# ================= CONFIG =================
OUTPUT_DIR = "./results_chat_finetune"
TOKENIZER_PATH = "Lyra_tokenizer.json"
PRETRAINED_MODEL_PATH = "./results_pretrain/final_pretrained_500m" # Path to your pretrained 500M model
MAX_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 10
DATASET_NAME = "daily_dialog" # Example conversational dataset

# --- CHECKPOINT AUTO-DETECT ---
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    print(f"🟢 Resuming from checkpoint: {last_checkpoint}")
else:
    print("🆕 No checkpoint found, starting fine-tuning")

# --- LOAD TOKENIZER ---
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH,
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)
print("Tokenizer vocab size:", tokenizer.vocab_size)

# --- LOAD MODEL ---
# Try to load the locally pretrained 500M model first
if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pretrained 500M model from {PRETRAINED_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL_PATH)
else:
    print(f"⚠️ Pretrained model not found at {PRETRAINED_MODEL_PATH}. Loading 'gpt2' base for testing.")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Resize embeddings if using custom tokenizer on standard gpt2
    model.resize_token_embeddings(len(tokenizer))

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- DATASET ---
print(f"Loading dataset: {DATASET_NAME}...")
raw_datasets = load_dataset(DATASET_NAME)
train_dataset = raw_datasets["train"]

# --- FORMATTING ---
def format_chat(example):
    # Daily Dialog format: lists of utterances
    dialog = example["dialog"]
    text = ""
    for i, utter in enumerate(dialog):
        role = "User" if i % 2 == 0 else "Assistant"
        text += f"{role}: {utter}\n"
    return text

def preprocess_function(examples, tokenizer, max_length):
    texts = [format_chat(ex) for ex in examples] # Wrap in list if needed, depending on dataset structure
    # For daily_dialog, 'examples' is a batch dict, so we iterate
    
    # Actually for 'map' with batched=True:
    # examples['dialog'] is a list of lists of strings
    
    formatted_texts = []
    for dialog in examples["dialog"]:
        text = ""
        for i, utter in enumerate(dialog):
            role = "User: " if i % 2 == 0 else "Assistant: "
            text += f"{role}{utter}<|endoftext|>"
        formatted_texts.append(text)

    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

process_func = partial(preprocess_function, tokenizer=tokenizer, max_length=MAX_LENGTH)

tokenized_dataset = train_dataset.map(
    process_func,
    batched=True,
    remove_columns=train_dataset.column_names
)

# --- TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    fp16=True,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    optim="adamw_bnb_8bit" # [NEW] Use 8-bit Adam
)

# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# --- TRAIN ---
trainer.train(resume_from_checkpoint=last_checkpoint)

# --- SAVE ---
trainer.save_model(f"{OUTPUT_DIR}/final_chat_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_chat_model")
print("Chat fine-tuning complete.")
