import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import MiniTransformer
from week5_reversal.data_generator import ReversalDataset, make_dataset, collate_fn, PAD, VOCAB_SIZE
from week5_reversal.train_week5 import get_masks, DEVICE

def overfit():
    # Generate exactly 32 examples
    train_data = make_dataset(32, seed=42)
    loader = DataLoader(ReversalDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    epochs = 1000  
    print("Overfitting on 32 examples...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        for src, tgt_in, labels in loader:
            src, tgt_in, labels = src.to(DEVICE), tgt_in.to(DEVICE), labels.to(DEVICE)
            src_pad_mask, causal_mask = get_masks(src, tgt_in)
            
            optimizer.zero_grad()
            logits = model(src, tgt_in, src_pad_mask, causal_mask)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
        if epoch % 30 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    print("Overfit complete. The loss should be near 0.000.")

if __name__ == "__main__":
    overfit()