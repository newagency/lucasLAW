import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets, DatasetDict

# Read Hugging Face token from environment variables
HF_TOKEN = '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'

if HF_TOKEN is None:
    raise ValueError("Hugging Face token is not set in the environment variable 'HF_TOKEN'.")

# 1. Load the base model and tokenizer
model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 since RTX 3090 does not support bfloat16
    device_map="auto",
    token=HF_TOKEN
)

# 2. Load the datasets
dataset1 = load_dataset("jihye-moon/LawQA-Ko", use_auth_token=HF_TOKEN)
dataset2 = load_dataset("zzunyang/LawQA_LawSee", use_auth_token=HF_TOKEN)

# Merge the datasets (for all splits like train, validation, etc.)
def merge_datasets(ds1, ds2):
    merged = {}
    for split in ds1.keys():
        if split in ds2:
            merged[split] = concatenate_datasets([ds1[split], ds2[split]])
        else:
            merged[split] = ds1[split]
    return merged

merged_datasets = merge_datasets(dataset1, dataset2)

# 3. Preprocess the data: Convert to prompt-response format
def preprocess_function(examples):
    # Combine question and answer for each example to form the model input format
    # Example: "Q: [Question]\nA: [Answer]"
    prompt = "Q: " + examples['question'] + "\nA: " + examples['answer']
    return {"text": prompt}

tokenized_datasets = merged_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=['question', 'answer']
)

# 4. Tokenize the text using the tokenizer
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_datasets = tokenized_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=4  # Number of processes for parallel processing (adjust based on CPU cores)
)

# 5. Change dataset format to PyTorch tensors
tokenized_datasets.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask']
)

# 6. Split the dataset into train and validation sets
# Here, we split 10% of the training data for validation
split_ratio = 0.3
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=split_ratio, seed=42)
tokenized_datasets = DatasetDict({
    "train": tokenized_datasets["train"],
    "validation": tokenized_datasets["test"]
})

# 7. Set up the data collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masking for Causal Language Modeling
)

# 8. Set up training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-llama3B-lawqa",
    overwrite_output_dir=True,
    num_train_epochs=3,                     # Number of epochs
    per_device_train_batch_size=2,          # Adjust based on GPU memory
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,                              # Use 16-bit floating point
    logging_dir='./logs',
    logging_steps=10,
)

# 9. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 10. Fine-tune the model
trainer.train()

# 11. Save the fine-tuned model
trainer.save_model("./fine-tuned-llama3B-lawqa")
tokenizer.save_pretrained("./fine-tuned-llama3B-lawqa")

