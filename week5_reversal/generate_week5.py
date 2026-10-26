import torch
from models.transformer import MiniTransformer
from week5_reversal.data_generator import make_dataset, BOS, EOS, PAD, VOCAB_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@torch.no_grad()
def greedy_decode(model, src, max_len=20):
    model.eval()
    B = src.size(0)
    
    # Start with just BOS
    generated = torch.full((B, 1), BOS, dtype=torch.long, device=DEVICE)
    
    # Track which sequences have finished (hit EOS)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    
    for _ in range(max_len):
        # Create masks for current generated sequence
        src_pad_mask = (src == PAD).unsqueeze(1).unsqueeze(2)
        T_tgt = generated.size(1)
        causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=DEVICE), diagonal=1)
        
        # Forward pass
        logits = model(src, generated, src_pad_mask, causal_mask)
        
        # Get the prediction for the LAST token in the sequence
        next_token_logits = logits[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1)
        
        # Append to generated
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        
        # Mark as finished if EOS is generated
        finished |= (next_tokens == EOS)
        if finished.all():
            break

    return generated

def test_generation():
    # Load model
    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    model.load_state_dict(torch.load("week5_model.pth", map_location=DEVICE))
    
    # Get 5 validation examples
    val_data = make_dataset(5, seed=99)
    
    print("Testing Greedy Decoding on 5 Examples:\n")
    
    for i, (src_seq, tgt_rev) in enumerate(val_data):
        src = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)
        
        # Generate output
        out = greedy_decode(model, src)[0].tolist()
        
        # Remove BOS and stop at EOS
        out = out[1:] # remove BOS
        if EOS in out:
            out = out[:out.index(EOS)]
            
        print(f"Example {i+1}:")
        print(f"  Source:     {list(src_seq)}")
        print(f"  Reference:  {list(tgt_rev)}")
        print(f"  Generated:  {out}")
        print(f"  Match:      {'YES' if out == list(tgt_rev) else 'NO'}\n")

if __name__ == "__main__":
    test_generation()