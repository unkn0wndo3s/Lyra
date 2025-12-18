import os
import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import IterableDataset
from transformers.trainer_utils import get_last_checkpoint

# --- CONFIG ---
OUTPUT_DIR = "./results_pretrain"
TOKENIZER_PATH = "Lyra_tokenizer.json"
CONFIG_PATH = "config_500m.json" # [NEW] Use the 500M config
MAX_LENGTH = 1024
BATCH_SIZE = 4 # [OPTIMIZED] Increased slightly, relying on DeepSpeed ZeRO-2
GRAD_ACCUM = 4 # [OPTIMIZED] Effective batch ~16
LEARNING_RATE = 1e-4 # [TUNED] Standard for training from scratch
MAX_STEPS = 50000 # [REAL] Increased for actual training
LOGGING_STEPS = 10

# --- CHECKPOINT AUTO-DETECT ---
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    print(f"🟢 Reprise depuis le checkpoint : {last_checkpoint}")
else:
    print("🆕 Aucun checkpoint trouvé, entraînement depuis zéro")

# --- LOAD TOKENIZER ---
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH,
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)
print("Tokenizer vocab size:", tokenizer.vocab_size)

# --- MODEL ---
if os.path.exists(CONFIG_PATH):
    config = AutoConfig.from_pretrained(CONFIG_PATH)
    print(f"Loaded configuration from {CONFIG_PATH}")
else:
    print(f"⚠️ Config file {CONFIG_PATH} not found. Falling back to GPT-2 Small defaults.")
    config = AutoConfig.for_model(
        "gpt2",
        vocab_size=tokenizer.vocab_size,
        n_positions=MAX_LENGTH,
        n_ctx=MAX_LENGTH,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

model = AutoModelForCausalLM.from_config(config)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# --- STREAMING DATASET ---
raw_dataset = load_dataset(
    "monology/pile-uncopyrighted",
    split="train",
    streaming=True
)

class TextOnlyIterable(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for example in self.dataset:
            text = example.get("text")
            if not text:
                continue

            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            yield tokenized

            count += 1
            if self.max_samples and count >= self.max_samples:
                break

train_dataset = TextOnlyIterable(
    raw_dataset,
    tokenizer,
    MAX_LENGTH,
    max_samples=None # [REAL] Unlimited for real training
)

# --- DATA COLLATOR ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --- TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True, # GPU support
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    save_steps=1000,
    save_total_limit=3,
    gradient_checkpointing=True, # [CRITICAL] Saves VRAM
    remove_unused_columns=False,
    report_to=[],
    max_steps=MAX_STEPS,
    dataloader_drop_last=True,
    optim="adamw_bnb_8bit" # [NEW] Use 8-bit Adam to save memory (replaces DeepSpeed)
)

# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# --- TRAIN (AUTO RESUME) ---
trainer.train(resume_from_checkpoint=last_checkpoint)

# --- SAVE FINAL ---
trainer.save_model(f"{OUTPUT_DIR}/final_pretrained_500m")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_pretrained_500m")
