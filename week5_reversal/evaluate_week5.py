import torch
from torch.utils.data import DataLoader
from models.transformer import MiniTransformer
from week5_reversal.data_generator import ReversalDataset, make_dataset, collate_fn, PAD, BOS, EOS, VOCAB_SIZE
from week5_reversal.train_week5 import get_masks, DEVICE
from week5_reversal.generate_week5 import greedy_decode

def evaluate():
    # Load the trained model
    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    model.load_state_dict(torch.load("week5_model.pth", map_location=DEVICE))
    model.eval()

    val_data = make_dataset(200, seed=99)
    loader = DataLoader(ReversalDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate_fn)

    exact_matches = 0
    total_gen_tokens_correct = 0
    total_gen_tokens = 0
    
    total_tf_tokens_correct = 0
    total_tf_tokens = 0

    with torch.no_grad():
        for src, tgt_in, labels in loader:
            src, tgt_in, labels = src.to(DEVICE), tgt_in.to(DEVICE), labels.to(DEVICE)
            
            # 1. Teacher-Forced Token Accuracy
            src_pad_mask, causal_mask = get_masks(src, tgt_in)
            logits = model(src, tgt_in, src_pad_mask, causal_mask)
            preds = logits.argmax(-1)
            
            # Only count non-PAD tokens
            mask = labels != PAD
            total_tf_tokens_correct += (preds[mask] == labels[mask]).sum().item()
            total_tf_tokens += mask.sum().item()

            # 2. Greedy Generation
            generated = greedy_decode(model, src) # [B, seq_len]
            
            for i in range(src.size(0)):
                # Process reference and generated
                ref = [t for t in labels[i].tolist() if t != PAD and t != EOS]
                gen = [t for t in generated[i].tolist() if t != BOS and t != PAD]
                if EOS in gen:
                    gen = gen[:gen.index(EOS)]
                
                # Exact Match
                if gen == ref:
                    exact_matches += 1
                
                # Generated Token Accuracy
                for j in range(min(len(ref), len(gen))):
                    if gen[j] == ref[j]:
                        total_gen_tokens_correct += 1
                total_gen_tokens += len(ref)

    print(f"Validation Set Size: {len(val_data)}")
    print(f"Exact-Sequence Accuracy (Generation): {exact_matches / len(val_data) * 100:.2f}%")
    print(f"Token Accuracy (Teacher Forced):      {total_tf_tokens_correct / total_tf_tokens * 100:.2f}%")
    print(f"Token Accuracy (Generated):           {total_gen_tokens_correct / total_gen_tokens * 100:.2f}%")
    print("\nNote: Teacher-forced token accuracy is higher because the model sees the correct previous tokens. Generation cascades errors.")

if __name__ == "__main__":
    evaluate()