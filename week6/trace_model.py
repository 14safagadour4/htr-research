import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print("Loading model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

print("\n" + "="*50)
print("1. PROCESSOR DETAILS")
print("="*50)
print(f"Image Processor Class: {processor.image_processor.__class__.__name__}")
print(f"Tokenizer Class: {processor.tokenizer.__class__.__name__}")
print(f"Special Tokens: {processor.tokenizer.special_tokens_map}")

print("\n" + "="*50)
print("2. MODEL CONFIG (Encoder/Decoder)")
print("="*50)
config = model.config
print(f"Encoder Class: {model.encoder.__class__.__name__}")
print(f"Decoder Class: {model.decoder.__class__.__name__}")
print(f"Encoder Image Size: {config.encoder.image_size}")
print(f"Encoder Patch Size: {config.encoder.patch_size}")
print(f"Encoder Hidden Size: {config.encoder.hidden_size}")
print(f"Decoder Vocab Size: {config.decoder.vocab_size}")
print(f"Decoder Max Position Embeddings: {config.decoder.max_position_embeddings}")

print("\n" + "="*50)
print("3. FORWARD PASS TENSOR SHAPES (Teacher Forcing)")
print("="*50)
# Create a dummy image tensor (Batch=1, Channels=3, Height=384, Width=384)
# 384 is standard, but TrOCR processor resizes to 384x384 by default.
dummy_pixel_values = torch.randn(1, 3, 384, 384)
# Create dummy labels (Batch=1, Sequence Length=10)
dummy_labels = torch.randint(0, config.decoder.vocab_size, (1, 10))

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
with torch.no_grad():

    outputs = model(pixel_values=dummy_pixel_values, labels=dummy_labels)

print(f"Input Image Shape: {dummy_pixel_values.shape}")
print(f"Encoder Last Hidden State Shape: {outputs.encoder_last_hidden_state.shape}")
print(f"Logits Shape: {outputs.logits.shape}  (Batch, Target Seq Len, Vocab Size)")
print(f"Loss: {outputs.loss.item():.4f}")