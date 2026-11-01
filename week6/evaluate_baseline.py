import torch
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
model.eval()

df = pd.read_csv("week6/data/frozen_manifest.csv")
predictions = []
references = []

print("Running inference on 100 samples...")
for index, row in df.iterrows():
    image = Image.open(row['image_path']).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate text (No labels provided!)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=50)
    
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predictions.append(pred_text)
    references.append(row['reference_text'])
    
    if index < 3: # Print first 3 examples to visually check
        print(f"\nRef:  {row['reference_text']}")
        print(f"Pred: {pred_text}")

# Calculate Metrics (Corpus level, as required by the plan)
corpus_cer = cer(references, predictions)
corpus_wer = wer(references, predictions)
exact_matches = sum([1 for p, r in zip(predictions, references) if p.strip() == r.strip()])
exact_match_acc = exact_matches / len(references)

print("\n" + "="*40)
print("BASELINE RESULTS (Without Fine-Tuning)")
print("="*40)
print(f"Corpus CER: {corpus_cer * 100:.2f}%")
print(f"Corpus WER: {corpus_wer * 100:.2f}%")
print(f"Exact-Match Accuracy: {exact_match_acc * 100:.2f}%")

df['prediction'] = predictions
df.to_csv("week6/data/baseline_predictions.csv", index=False)
print("\nSaved predictions to week6/data/baseline_predictions.csv")