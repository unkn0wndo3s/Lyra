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

# --- CLASS DEFINITIONS MUST BE AT TOP LEVEL FOR PICKLING ---
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

if __name__ == "__main__":
    # --- CONFIG ---
    OUTPUT_DIR = "./results_pretrain"
    TOKENIZER_PATH = "Lyra_tokenizer.json"
    # CONFIG_PATH = "config_355m.json" # [DISABLED] Using default GPT-2 (124M)
    MAX_LENGTH = 1024 # [RESTORE] Standard context
    BATCH_SIZE = 8 # [FAST] 124M is small, can handle larger batches
    GRAD_ACCUM = 4 
    LEARNING_RATE = 5e-4 # [TUNED] Higher LR for smaller model
    MAX_STEPS = 50000 
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
    # [SPEED] Using 124M (GPT-2 Small) as requested
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
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters (GPT-2 Small 124M).")

    # --- DATASET (LOCAL/CACHED) ---
    # [FIX] Switched to WikiText-103 (download once, train fast) to avoid network streaming crashes
    print("Downloading/Loading WikiText-103 dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    print("Tokenizing dataset locally (this handles speed)...")
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # [FIX] Use standard dataset, not streaming iterable
    train_dataset = tokenized_dataset

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
        gradient_checkpointing=True, 
        remove_unused_columns=False,
        report_to=[],
        max_steps=MAX_STEPS,
        dataloader_drop_last=True,
        optim="paged_adamw_8bit", # [SPEED] Paged optimizer invokes explicit CPU offload only if needed
        dataloader_num_workers=0, # [FIX] 0 workers on Windows prevents the "client closed" and pickling errors
        dataloader_pin_memory=True # [SPEED] Faster transfer to GPU
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
