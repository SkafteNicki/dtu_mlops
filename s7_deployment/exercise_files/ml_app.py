from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.post("/caption/")
async def caption(data: UploadFile = File(...)):
    """Generate a caption for an image."""
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
