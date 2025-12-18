import os
import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM, # Keeping CausalLM, but structure data for Q&A style
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling # Used for padding/masking
)
from transformers.trainer_utils import get_last_checkpoint
from functools import partial

# ================= CONFIG (Q&A Fine-Tuning) =================
OUTPUT_DIR = "./results_qna_finetune"
TOKENIZER_PATH = "Lyra_tokenizer.json" # Assumes tokenizer from previous script exists
MAX_LENGTH = 512                      # Shorter length is suitable for Q&A context/answer
BATCH_SIZE = 4                        # Increased batch size for efficiency
GRAD_ACCUM = 4                        # Total effective batch size: BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE = 2e-5                  # Lower LR for fine-tuning
NUM_TRAIN_EPOCHS = 3.0                # Fine-tuning uses epochs, not just max_steps
LOGGING_STEPS = 50
QNA_DATASET_NAME = "squad"            # Dedicated Q&A dataset for better results

# --- CHECKPOINT AUTO-DETECT ---
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    print(f"🟢 Resuming from checkpoint: {last_checkpoint}")
else:
    print("🆕 No checkpoint found, starting fine-tuning from zero (or pre-trained base)")

# --- LOAD TOKENIZER ---
# Ensure special tokens match the pre-training configuration
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH,
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>"
)
print("Tokenizer vocab size:", tokenizer.vocab_size)

# --- MODEL (Load Pre-trained or Start Fresh) ---
# For a true single-pass Q&A setup, you would ideally load the model trained
# in the *first* script here. For simplicity, we re-initialize the same config.
# If a pre-trained model is available, replace AutoModelForCausalLM.from_config()
# with AutoModelForCausalLM.from_pretrained("<PATH_TO_PRETRAINED_MODEL>").

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

# Start from scratch if no checkpoint, or load if one exists
model = AutoModelForCausalLM.from_config(config)

if last_checkpoint:
    # Attempt to load model weights from the last checkpoint
    model = AutoModelForCausalLM.from_pretrained(last_checkpoint)
    print(f"Model weights loaded from: {last_checkpoint}")
else:
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# --- Q&A DATASET (Automatic Download) ---
print(f"Downloading and preparing Q&A dataset: {QNA_DATASET_NAME}...")
raw_datasets = load_dataset(QNA_DATASET_NAME)

# Use only the training split for fine-tuning
train_dataset = raw_datasets["train"]

# --- DATA PROCESSING FUNCTION (Q&A FORMAT) ---
def format_qa_example(examples, tokenizer, max_length):
    """
    Formats SQuAD examples into a prompt/response pair suitable for Causal Language Modeling (CLM).
    Format: "<|endoftext|>Question: [Question] Context: [Context] Answer: [Answer]<|endoftext|>"
    """
    prompts = []
    responses = []
    
    # Iterate through batch
    for question, context, answer_dict in zip(
        examples["question"], examples["context"], examples["answers"]
    ):
        # SQuAD may have multiple answers, we use the first one
        answer = answer_dict["text"][0] if answer_dict["text"] else ""
        
        # Create the full text sequence
        full_text = (
            f"<|endoftext|>Question: {question} Context: {context} Answer: {answer}<|endoftext|>"
        )
        responses.append(full_text)

    # Tokenize the full sequence
    tokenized_output = tokenizer(
        responses,
        truncation=True,
        max_length=max_length,
        padding="max_length", # Using padding for batching efficiency
        return_tensors="pt"
    )
    
    # CLM training requires 'labels' to be the same as 'input_ids' for loss calculation
    tokenized_output["labels"] = tokenized_output["input_ids"].clone()
    
    return tokenized_output

# Apply formatting and tokenization
# Using functools.partial to pass tokenizer and max_length to the map function
formatting_func = partial(format_qa_example, tokenizer=tokenizer, max_length=MAX_LENGTH)

tokenized_train_dataset = train_dataset.map(
    formatting_func,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print(f"Tokenized dataset size: {len(tokenized_train_dataset)} samples")

# --- DATA COLLATOR ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # Crucial for Causal Language Modeling
)

# --- TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    fp16=torch.cuda.is_available(),  # Enable FP16 if CUDA is available
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    save_strategy="epoch", # Save checkpoint after each epoch
    save_total_limit=2,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to=[],
    dataloader_drop_last=True
)

# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator
)

# --- TRAIN (AUTO RESUME) ---
print("Starting Q&A Fine-Tuning...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# --- SAVE FINAL ---
final_model_path = f"{OUTPUT_DIR}/final_qna_finetuned_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\n🎉 Q&A Fine-tuning complete. Model saved to: {final_model_path}")