import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", local_files_only=True).to(device)

# Put model in TRAINING mode
model.train() 

# Set decoder start token (needed for calculating loss during training)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

print("Loading 1 training sample from the frozen manifest...")
df = pd.read_csv("week6/data/frozen_manifest.csv")
image = Image.open(df.iloc[0]['image_path']).convert("RGB")
text = df.iloc[0]['reference_text']

# Prepare inputs
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
labels = processor.tokenizer(text, padding="max_length", max_length=20, return_tensors="pt").input_ids.to(device)

print(f"Image shape: {pixel_values.shape}")
print(f"Label shape: {labels.shape}")

print("Running forward pass and backward pass...")
outputs = model(pixel_values=pixel_values, labels=labels)
loss = outputs.loss
loss.backward()

print(f"Loss: {loss.item():.4f}")
print("SUCCESS: Backward pass completed without Out-Of-Memory (OOM) errors! You are ready for Week 7.")