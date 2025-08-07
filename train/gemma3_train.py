import io
from typing import Any, cast
import requests
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import IterableDataset, Features, load_dataset
import datasets
from PIL import Image
import numpy as np


def load_image(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image

def image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
)
model.config.use_cache = False  # Disable caching for training

processor = AutoProcessor.from_pretrained(model_id, padding_side="right")
processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
processor.tokenizer.padding_side = "right"

train_data = load_dataset("flaviagiammarino/path-vqa", split="train")
train_data = train_data.select(range(1000))
print(train_data[0])


def format_example(example):
    return {
        "images": example["image"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["question"]},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["answer"]}
                ]
            }
        ]
    }
    
train_ds = train_data.map(format_example)

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # print("collate_fn pixel_values", batch["pixel_values"].shape)
    # print("collate_fn input_ids", batch["input_ids"].shape)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.image_token_id] = -100
    batch["labels"] = labels

    return batch

# Set up LoRA configuration for causal language modeling
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Define training arguments
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=25,
    #report_to="tensorboard",
    group_by_length=False,
    remove_unused_columns=False,
    dataset_kwargs = {"skip_prepare_dataset": True},
    gradient_checkpointing_kwargs = dict(use_reentrant=False),
)

# Create the SFTTrainer with LoRA parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=cast(Any, train_ds),
    peft_config=lora_config,
    args=training_args,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)

trainer.train()