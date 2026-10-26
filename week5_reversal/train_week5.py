
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import MiniTransformer

try:
    from week5_reversal.data_generator import ReversalDataset, make_dataset, collate_fn, PAD, VOCAB_SIZE
except ImportError:
    from week5_reversal.data_generator import ReversalDataset, make_dataset, collate_fn, PAD, VOCAB_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_masks(src, tgt_in):
    # Encoder padding mask
    src_pad_mask = (src == PAD).unsqueeze(1).unsqueeze(2)
    
    # Decoder causal mask
    T_tgt = tgt_in.size(1)
    causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=DEVICE), diagonal=1)
    
    return src_pad_mask, causal_mask

def train():
    # 1. Data
    train_data = make_dataset(2000, seed=42)
    val_data = make_dataset(200, seed=99)
    
    train_loader = DataLoader(ReversalDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ReversalDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. Model
    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    epochs = 30
    print(f"Training on {DEVICE}...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for src, tgt_in, labels in train_loader:
            src, tgt_in, labels = src.to(DEVICE), tgt_in.to(DEVICE), labels.to(DEVICE)
            src_pad_mask, causal_mask = get_masks(src, tgt_in)
            
            optimizer.zero_grad()
            logits = model(src, tgt_in, src_pad_mask, causal_mask)
            
            # Flatten logits and labels for CrossEntropyLoss
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), "week5_model.pth")
    print("Model saved to week5_model.pth")

if __name__ == "__main__":
    train()