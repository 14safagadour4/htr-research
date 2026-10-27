import torch
from models.transformer import MiniTransformer
from week5_reversal.data_generator import ReversalDataset, make_dataset, PAD, BOS, EOS, VOCAB_SIZE
from week5_reversal.train_week5 import DEVICE
from week5_reversal.generate_week5 import greedy_decode

def check():
    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    model.load_state_dict(torch.load("overfit_model.pth", map_location=DEVICE))
    model.eval()
    
    data = make_dataset(32, seed=42)
    correct = 0
    for src_seq, tgt_rev in data:
        src = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)
        out = greedy_decode(model, src)[0].tolist()[1:] # skip BOS
        if EOS in out: out = out[:out.index(EOS)]
        if out == list(tgt_rev): correct += 1
        
    print(f"Overfit Exact-Match: {correct}/32 ({correct/32*100:.1f}%)")

if __name__ == "__main__":
    check()