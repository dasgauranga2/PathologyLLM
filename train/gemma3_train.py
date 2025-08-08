from typing import Any, cast
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import os

# model id
model_id = "google/gemma-3-4b-it"
# load model
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
)
model.config.use_cache = False  # disable caching for training

# use wandb for monitoring metrics with project name 'VLM_DPO'
os.environ["WANDB_PROJECT"] = "PATH_VQA"

# load model processor
processor = AutoProcessor.from_pretrained(model_id, padding_side="right")
processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
processor.tokenizer.padding_side = "right"

# load training data
train_data = load_dataset("flaviagiammarino/path-vqa", split="train")
#train_data = train_data.select(range(1000))
print(train_data[0])

# format each data sample into a format for gemma 3 processor
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

# format the dataset
train_ds = train_data.map(format_example)

# collate function
# combine list of data points to create a batch of input data
def collate_fn(examples):
    # process the text data
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    # list of images
    images = [example["images"] for example in examples]

    # process the text and input images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # labels are the target outputs
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.image_token_id] = -100
    batch["labels"] = labels

    return batch

# setup LoRA config
# important if GPU resources are limited
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# training configuration
training_args = SFTConfig(
    output_dir="checkpoint/",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2, # keep only the latest one
    #save_steps=25,
    report_to="wandb",
    run_name="gemma3",
    group_by_length=False,
    remove_unused_columns=False,
    dataset_kwargs = {"skip_prepare_dataset": True},
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = dict(use_reentrant=False),
)

# trainer class
trainer = SFTTrainer(
    model=model,
    train_dataset=cast(Any, train_ds),
    peft_config=lora_config,
    args=training_args,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)

# start training
trainer.train()