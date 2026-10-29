import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

print("Model and processor loaded successfully.")
print(f"Model class: {model.__class__.__name__}")
print(f"Encoder class: {model.encoder.__class__.__name__}")
print(f"Decoder class: {model.decoder.__class__.__name__}")