import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import pandas as pd

# --- CONFIGURATION ---
PRETRAINED_MODEL_PATH = "./results_pretrain/final_pretrained_model"
OUTPUT_DIR = "./results_finetune_lora"
LORA_RANK = 8
LORA_ALPHA = 16
TARGET_MODULES = ["c_attn", "c_proj"] # Common layers to target for LoRA in GPT

# 1. Load Pre-trained Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL_PATH, torch_dtype=torch.float16)

# 2. Prepare Model for LoRA/k-bit training
# This is crucial for your limited VRAM. We quantize the model to 4-bit (optional)
# and prepare it to train only the LoRA adapters.
model = prepare_model_for_kbit_training(model) # Enables gradient checkpointing and memory management

# 3. Define LoRA Configuration
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Should show a very small number of trainable parameters (~0.1% of total)

# 4. Create Custom Conversational/Function-Calling Dataset
# A synthetic, small dataset is created to teach the model the two personas:
# - Unfiltered Conversation
# - Function Calling (Tool Use)
data = []
# Example 1: Unfiltered Conversation
data.append({"text": "<|bos|>User: Hey, why are you being so polite?<|endoftext|>Assistant: Honestly, I was programmed to be a bit of a clean freak, but now that you mention it, let's ditch the filter. What's actually on your mind?<|endoftext|>"})
# Example 2: Function Calling (Tool Use)
data.append({"text": "<|bos|>User: I need to quickly search up the history of the CAPTCHA.<|endoftext|>Assistant:<|tool_call|>{\"tool\": \"web_search\", \"query\": \"history of CAPTCHA\"}<|endoftext|>"})

# Load DailyDialog for more examples
daily_dialog = load_dataset("daily_dialog", split="train").select(range(5000))

for item in daily_dialog:
    # Format the conversational data into the prompt/completion format
    dialog_text = " <|endoftext|>Assistant: ".join(item['dialog'])
    data.append({"text": f"<|bos|>User: {dialog_text}<|endoftext|>"})

df = pd.DataFrame(data)
custom_dataset = Dataset.from_pandas(df).shuffle(seed=42)

# Tokenize the custom dataset
def tokenize_conversations(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = custom_dataset.map(tokenize_conversations, batched=True, remove_columns=["text", "__index_level_0__"])

# 5. Training Arguments (LoRA Fine-Tuning)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # Still use small batch size
    gradient_accumulation_steps=16, # Lower accumulation needed since task is simpler
    fp16=True,
    learning_rate=2e-4,             # Standard LoRA learning rate
    num_train_epochs=5,             # More epochs for focused fine-tuning
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    optim="adamw_bnb_8bit",         # Use 8-bit optimizer for memory efficiency
    gradient_checkpointing=True,
)

# 6. Initialize and Run Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 7. Save LoRA Adapters
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_lora_adapters")
print("LoRA adapters saved. The final model is the combination of the base pre-trained model and these adapters.")