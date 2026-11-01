import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from datasets import load_dataset


print("Loading IAM dataset (Line-level for this pilot)...")
# Teklia/IAM-line is a standard, clean HuggingFace dataset for IAM
dataset = load_dataset("Teklia/IAM-line", split="validation") 

# Freeze the first 100 samples to keep evaluation fast this week
frozen_subset = dataset.select(range(100))

os.makedirs("week6/data", exist_ok=True)
manifest = []

print("Saving 100 images to week6/data/...")
for i, example in enumerate(frozen_subset):
    # Save the image locally
    img_path = f"week6/data/sample_{i}.png"
    example['image'].save(img_path)
    
    # Add to manifest
    manifest.append({
        "sample_id": i,
        "image_path": img_path,
        "reference_text": example['text']
    })

df = pd.DataFrame(manifest)
df.to_csv("week6/data/frozen_manifest.csv", index=False)
print("SUCCESS: Saved frozen_manifest.csv with 100 samples.")