import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# 1. Authenticate (ensure you've accepted the Gemma license)
# HF_TOKEN = os.environ.get("HF_TOKEN")
# login(token=HF_TOKEN)

model_id = "google/gemma-3-1b-it"  # adjust variant as needed

# 2. Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. Load dataset
dataset = load_dataset("meta-math/MetaMathQA-40K", split="train")
  # replace with your data source
dataset = dataset.select(range(1000))

# 4. Preprocess dataset to prompt-completion format (if needed)
def preprocess_fn(example):
    return {"prompt": example["query"], "completion": example["response"]}

dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)

# 5. Configure training
training_args = SFTConfig(
    output_dir="./gemma3_sft",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    packing=False,  # enable if your data benefits from sequence packing
)

# 6. Initialize trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# 7. Train
trainer.train()
trainer.save_model("./gemma3_sft_model")