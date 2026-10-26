"""
Week 4 — Ablation that demonstrates why each component is needed.

Trains four variants of MiniTransformer for 200 steps on the 32-example
sequence-reversal set (the same overfit set Week 5 uses as its first
diagnostic). Each variant removes exactly one component:

    full         — all four components present
    no_pos       — positional encoding removed
    no_ffn       — FFN replaced by identity
    no_residual  — the residual additions removed

"""
from __future__ import annotations

import random
import torch
import torch.nn as nn

from models.transformer import MiniTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD, BOS, EOS = 10, 11, 12
VOCAB_SIZE = 13


# ─────────────────────────────────────────────────────────────────────────
# Data — 32 unique reversal examples
# ─────────────────────────────────────────────────────────────────────────

def make_examples(n=32, seed=42):
    random.seed(seed)
    examples, seen = [], set()
    while len(examples) < n:
        L = random.randint(3, 12)
        seq = tuple(random.randint(0, 9) for _ in range(L))
        if seq in seen:
            continue
        seen.add(seq)
        examples.append((seq, tuple(reversed(seq))))
    return examples


def collate(batch):
    """Pad to the batch maximum; build decoder input and labels."""
    srcs, tgts = zip(*batch)
    T_src = max(len(s) for s in srcs)
    T_tgt = max(len(t) for t in tgts) + 2          # BOS + EOS

    src_ids = torch.full((len(srcs), T_src), PAD, dtype=torch.long)
    tgt_in  = torch.full((len(tgts), T_tgt), PAD, dtype=torch.long)
    tgt_lab = torch.full((len(tgts), T_tgt), PAD, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_ids[i, :len(s)] = torch.tensor(s)
        tgt_in[i, 0] = BOS
        tgt_in[i, 1:1 + len(t)] = torch.tensor(t)
        tgt_lab[i, :len(t)] = torch.tensor(t)
        tgt_lab[i, len(t)] = EOS
    return src_ids, tgt_in, tgt_lab


# ─────────────────────────────────────────────────────────────────────────
# Variants — remove one component at a time
# ─────────────────────────────────────────────────────────────────────────

def make_variant(kind: str) -> MiniTransformer:
    m = MiniTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_heads=4, d_ff=256)

    if kind == "full":
        return m

    if kind == "no_pos":
        m.pos_enc = nn.Identity()
        return m

    if kind == "no_ffn":
        for blk in (m.encoder, m.decoder):
            blk.ffn = nn.Identity()
        return m

    if kind == "no_residual":
        def forward_no_residual(self, x, encoder_out=None,
                                self_mask=None, cross_mask=None):
            h = self.norm1(x)
            attn_out, _ = self.self_attn(h, h, h, mask=self_mask)
            x = self.dropout(attn_out)            # no residual
            if self.is_decoder and encoder_out is not None:
                h = self.norm2(x)
                cross_out, _ = self.cross_attn(h, encoder_out, encoder_out,
                                               mask=cross_mask)
                x = self.dropout(cross_out)       # no residual
            h = self.norm3(x) if self.is_decoder else self.norm2(x)
            ffn_out = self.ffn(h)
            x = self.dropout(ffn_out)             # no residual
            return x

        for blk in (m.encoder, m.decoder):
            blk.forward = forward_no_residual.__get__(blk, type(blk))
        return m

    raise ValueError(f"unknown variant: {kind}")


# ─────────────────────────────────────────────────────────────────────────
# Train + evaluate
# ─────────────────────────────────────────────────────────────────────────

def build_masks(src_ids, tgt_in):
    src_pad = (src_ids == PAD).view(-1, 1, 1, src_ids.size(1))
    T_tgt = tgt_in.size(1)
    causal = torch.triu(
        torch.ones(T_tgt, T_tgt, dtype=torch.bool, device=DEVICE), diagonal=1
    ).view(1, 1, T_tgt, T_tgt)
    return src_pad, causal


@torch.no_grad()
def exact_match(model, src_ids, tgt_in, tgt_lab, src_pad, causal):
    model.eval()
    logits = model(src_ids, tgt_in, src_pad_mask=src_pad, causal_mask=causal)
    preds = logits.argmax(-1)
    correct = 0
    for i in range(preds.size(0)):
        gen = []
        # Start at 0, because position 0 predicts the FIRST target token
        for t in range(preds.size(1)):
            tok = preds[i, t].item()
            if tok == EOS:
                break
            if tok != PAD: # Ignore padding predictions
                gen.append(tok)
                
        ref = [tok for tok in tgt_lab[i].tolist() if tok != PAD and tok != EOS]
        if gen == ref:
            correct += 1
    return correct / preds.size(0)


def run_variant(kind, src_ids, tgt_in, tgt_lab, src_pad, causal,
                steps=200, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = make_variant(kind).to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    final_loss = float("nan")
    for step in range(steps):
        logits = model(src_ids, tgt_in, src_pad_mask=src_pad, causal_mask=causal)
        loss = crit(logits.reshape(-1, VOCAB_SIZE), tgt_lab.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    em = exact_match(model, src_ids, tgt_in, tgt_lab, src_pad, causal)
    return final_loss, em


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("WEEK 4 ABLATION — why each Transformer component is needed")
    print(f"device = {DEVICE}")
    print("=" * 64 + "\n")

    examples = make_examples(32)
    src_ids, tgt_in, tgt_lab = collate(examples)
    src_ids, tgt_in, tgt_lab = (
        src_ids.to(DEVICE), tgt_in.to(DEVICE), tgt_lab.to(DEVICE)
    )
    src_pad, causal = build_masks(src_ids, tgt_in)

    results = {}
    for kind in ["full", "no_pos", "no_ffn", "no_residual"]:
        loss, em = run_variant(kind, src_ids, tgt_in, tgt_lab, src_pad, causal, steps=600)
        results[kind] = (loss, em)
        print(f"{kind:<13s} final loss = {loss:7.4f}   "
              f"exact-match (32) = {em * 100:5.1f}%")

    print("\nSummary")
    print("-" * 40)
    for kind, (loss, em) in results.items():
        print(f"  {kind:<13s} → {em * 100:5.1f}%")
    print("=" * 64)

if __name__ == "__main__":
    main()