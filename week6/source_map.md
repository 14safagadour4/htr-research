# TrOCR Source Map

## 1. The Processor (`TrOCRProcessor`)
- **Image Processor**: `ViTImageProcessor` (Resizes input image to 384x384, normalizes pixel values).
- **Tokenizer**: `RobertaTokenizerFast`
- **Special Tokens**: `{'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}`
- **Role**: Prepares raw RGB crops and target text into PyTorch tensors.

## 2. The Encoder (`ViTModel`)
- **Role**: Reads the image and builds visual context.
- **Input Resolution**: 384 x 384
- **Patch Size**: 16 x 16
- **Hidden Size**: 768
- **Tensor Flow**: 
  - Input image: `[1, 3, 384, 384]`
  - Patch division: 24x24 grid = 576 patches (+ 1 CLS token = 577 total).
  - Encoder Output: `[1, 577, 768]`

## 3. The Decoder (`TrOCRForCausalLM`)
- **Role**: Generates the text autoregressively.
- **Vocab Size**: 50,265
- **Max Position Embeddings**: 512
- **Tensor Flow**: 
  - Takes text token IDs (Batch, Seq Len).
  - Applies causal self-attention and cross-attention to the encoder's 577 patches.
  - Logits Output: `[1, 10, 50265]` (Batch, Target Seq Len, Vocab Size).

## 4. Generation (`model.generate`)
- **Role**: During inference (no labels provided), the model starts with the `<s>` (BOS) token, generates the next token, appends it, and repeats until it predicts `</s>` (EOS) or hits the length limit.