import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from models.transformer import MiniTransformer
from week5_reversal.data_generator import ReversalDataset, make_dataset, collate_fn, PAD, BOS, EOS, VOCAB_SIZE
from week5_reversal.train_week5 import get_masks, DEVICE
from week5_reversal.generate_week5 import greedy_decode

def evaluate_by_group():
    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    model.load_state_dict(torch.load("week5_model.pth", map_location=DEVICE))
    model.eval()

    val_data = make_dataset(200, seed=99)
    
    results_by_length = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_repeat = {"with_repeats": {"correct": 0, "total": 0}, 
                         "no_repeats": {"correct": 0, "total": 0}}

    for src_seq, tgt_rev in val_data:
        src = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)
        
        # Generate
        out = greedy_decode(model, src)[0].tolist()[1:] 
        if EOS in out: out = out[:out.index(EOS)]
        
        is_correct = (out == list(tgt_rev))
        L = len(src_seq)
        
        # Group by length
        results_by_length[L]["total"] += 1
        if is_correct: results_by_length[L]["correct"] += 1
        
        # Group by repeated tokens
        has_repeats = len(set(src_seq)) != len(src_seq)
        group = "with_repeats" if has_repeats else "no_repeats"
        results_by_repeat[group]["total"] += 1
        if is_correct: results_by_repeat[group]["correct"] += 1

    print("Accuracy by Sequence Length:")
    for L in sorted(results_by_length.keys()):
        c = results_by_length[L]["correct"]
        t = results_by_length[L]["total"]
        print(f"  Length {L:2d}: {c/t*100:5.1f}% ({c}/{t})")

    print("\nAccuracy by Token Repetition:")
    for group, data in results_by_repeat.items():
        c = data["correct"]
        t = data["total"]
        print(f"  {group:15s}: {c/t*100:5.1f}% ({c}/{t})")

if __name__ == "__main__":
    evaluate_by_group()