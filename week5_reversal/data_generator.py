
import random
import torch
from torch.utils.data import Dataset

PAD, BOS, EOS = 10, 11, 12
VOCAB_SIZE = 13

def make_dataset(n_samples, min_len=3, max_len=12, seed=42):
    random.seed(seed)
    data = []
    seen = set()
    while len(data) < n_samples:
        L = random.randint(min_len, max_len)
        seq = tuple(random.randint(0, 9) for _ in range(L))
        if seq in seen:
            continue
        seen.add(seq)
        data.append((seq, tuple(reversed(seq))))
    return data

class ReversalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt_rev = self.data[idx]
        
        tgt_in = [BOS] + list(tgt_rev)
        label = list(tgt_rev) + [EOS]
        
        return torch.tensor(src), torch.tensor(tgt_in), torch.tensor(label)

def collate_fn(batch):
    """Pads batches to the maximum length in the batch."""
    srcs, tgt_ins, labels = zip(*batch)
    
    src_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD)
    tgt_in_padded = torch.nn.utils.rnn.pad_sequence(tgt_ins, batch_first=True, padding_value=PAD)
    label_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD)
    
    return src_padded, tgt_in_padded, label_padded

if __name__ == "__main__":
    
    train_data = make_dataset(2000, seed=42)
    val_data = make_dataset(200, seed=99)
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    ds = ReversalDataset(train_data)
    src, tgt_in, lbl = ds[0]
    print(f"\nExample 0:")
    print(f"Source:   {src.tolist()}")
    print(f"Dec Input:{tgt_in.tolist()}")
    print(f"Label:    {lbl.tolist()}")