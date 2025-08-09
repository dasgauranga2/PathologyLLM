import io
from typing import Optional
from PIL import Image
import requests
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# INSTRUCTIONS
# Script to deploy Pytorch model on EC2 instance using FastAPI
# 1. Run the server using uvicorn server:app --host 0.0.0.0 --port 8000
# 2. Test with image URL - curl -X POST "http://<HOST>:8000/v1/generate" -F 'prompt=Describe this image in detail.' -F 'image_url=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg'
# 3. Test with image upload - curl -X POST "http://<HOST>:8000/v1/generate" -F 'prompt=Describe this image in detail.' -F 'image_file=@bee.jpg'

# model to use
MODEL_ID = "google/gemma-3-4b-it"

# creates a FastAPI application
# to create the web server
app = FastAPI(title="Gemma 3 VLM API")

# select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto"
).eval()
# load model processor
processor = AutoProcessor.from_pretrained(MODEL_ID)

# create an endpoint to check if server is deployed properly
@app.get("/health")
def health():
    return {"status": "ok"}

def _load_image_from_url(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

def _load_image_from_upload(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid uploaded image: {e}")

# create a POST endpoint called /v1/generate
@app.post("/v1/generate")
def generate(
    prompt: str = Form(..., description="User question or instruction"),
    image_url: Optional[str] = Form(None, description="HTTP(S) URL to an image"),
    image_file: Optional[UploadFile] = File(None, description="Uploaded image file")
):  
    # if image is not given raise an exception
    if not image_url and not image_file:
        raise HTTPException(status_code=400, detail="Provide either image_url or image_file.")

    # build Gemma-3 chat template
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": []}
    ]

    # put the image
    if image_url:
        # Gemma-3 processors accept URL strings directly for image content
        messages[1]["content"].append({"type": "image", "image": image_url})
    else:
        pil_img = _load_image_from_upload(image_file)
        # For uploads, pass a PIL.Image to the processor
        messages[1]["content"].append({"type": "image", "image": pil_img})

    # put the prompt
    messages[1]["content"].append({"type": "text", "text": prompt})

    # Tokenize with chat template (bf16 to match your example)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        # generate the output
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )
        # drop the prompt tokens to return only generated text
        gen = outputs[0][input_len:]

    # decode the text
    text = processor.decode(gen, skip_special_tokens=True)

    # return json result
    return JSONResponse({"answer": text})