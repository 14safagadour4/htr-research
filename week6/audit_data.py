from datasets import load_dataset
from PIL import Image

print("Downloading/loading IAM dataset (line-level)...")

dataset = load_dataset("Teklia/IAM-line", split="train")

print(f"\nTotal training samples: {len(dataset)}")
print("Dataset structure:", dataset.features)

# Inspect the first 3 samples
for i in range(3):
    example = dataset[i]
    print(f"\n--- Sample {i} ---")
    print(f"Text: {example['text']}")
    print(f"Image size: {example['image'].size}")
    
    # Save the first image locally so you can view it
    if i == 0:
        example['image'].save("week6/sample_iam.png")
        print("Saved first sample image to week6/sample_iam.png")