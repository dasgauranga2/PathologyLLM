import io
from typing import Any, cast
import requests
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import IterableDataset, Features
import datasets
from PIL import Image
import numpy as np


def load_image(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image

def image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def main():

    model_id = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    )
    model.config.use_cache = False  # Disable caching for training

    processor = AutoProcessor.from_pretrained(model_id, padding_side="right")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Use eos token as pad token
    processor.tokenizer.padding_side = "right"

    def train_iterable_gen():
        N_IMAGES = 1
        image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg").resize((896, 896))
        images = np.array([image] * N_IMAGES)
        print("IMAGES SHAPE", images.shape)
        yield {
                "images": images,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image" } for _ in range(images.shape[0])]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "duck" }]
                    }
                ]
            }
    train_ds = IterableDataset.from_generator(
         train_iterable_gen,
        features=Features({
            'images': [datasets.Image(mode=None, decode=True, id=None)],
            'messages': [{'content': [{'text': datasets.Value(dtype='string', id=None), 'type': datasets.Value(dtype='string', id=None) }], 'role': datasets.Value(dtype='string', id=None)}]
            } )
    )

    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        print("collate_fn pixel_values", batch["pixel_values"].shape)
        print("collate_fn input_ids", batch["input_ids"].shape)

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
        max_steps=1
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

    print("Training model...")
    trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    main()