import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.transformer import MiniTransformer
from week5_reversal.data_generator import ReversalDataset, make_dataset, collate_fn, PAD, BOS, EOS, VOCAB_SIZE
from week5_reversal.generate_week5 import greedy_decode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_masks(src, tgt_in):
    src_pad_mask = (src == PAD).unsqueeze(1).unsqueeze(2)
    T_tgt = tgt_in.size(1)
    causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=DEVICE), diagonal=1)
    return src_pad_mask, causal_mask

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    exact_matches = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    for src, tgt_in, labels in loader:
        src, tgt_in, labels = src.to(DEVICE), tgt_in.to(DEVICE), labels.to(DEVICE)
        src_pad_mask, causal_mask = get_masks(src, tgt_in)
        
        # Calculate validation loss
        logits = model(src, tgt_in, src_pad_mask, causal_mask)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
        total_loss += loss.item()
        
        # Calculate exact match using greedy decoding
        generated = greedy_decode(model, src)
        for i in range(src.size(0)):
            ref = [t for t in labels[i].tolist() if t != PAD and t != EOS]
            gen = [t for t in generated[i].tolist() if t != BOS and t != PAD]
            if EOS in gen:
                gen = gen[:gen.index(EOS)]
            if gen == ref:
                exact_matches += 1
            total_samples += 1
            
    return total_loss / len(loader), exact_matches / total_samples

def train():
    train_data = make_dataset(2000, seed=42)
    val_data = make_dataset(200, seed=99)
    
    train_loader = DataLoader(ReversalDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ReversalDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256, max_len=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    epochs = 30
    
    # Store real metrics
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Training on {DEVICE}...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for src, tgt_in, labels in train_loader:
            src, tgt_in, labels = src.to(DEVICE), tgt_in.to(DEVICE), labels.to(DEVICE)
            src_pad_mask, causal_mask = get_masks(src, tgt_in)
            
            optimizer.zero_grad()
            logits = model(src, tgt_in, src_pad_mask, causal_mask)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Calculate real validation metrics
        train_loss_avg = total_loss / len(train_loader)
        val_loss_avg, val_acc = evaluate(model, val_loader)
        
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc*100:.2f}%")

    torch.save(model.state_dict(), "week5_model.pth")
    print("Model saved to week5_model.pth")

    # --- PLOT REAL DATA ---
    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=3)
    axes[0].plot(epochs_range, history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[1].plot(epochs_range, [acc * 100 for acc in history['val_acc']], label='Validation Exact-Match', color='green', marker='s', markersize=3)
    axes[1].set_title('Validation Exact-Sequence Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week5_reversal/combined_curves.png', dpi=150)
    print("Saved combined_curves.png with REAL data.")

if __name__ == "__main__":
    train()