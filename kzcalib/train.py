"""Fine-tune a transformer POS tagger on the canonical split and dump logits.

Manual training loop (no Trainer) so that per-token logits on dev/test are a
first-class output. First-subword pooling: a word's label is attached to its
first subword; other subwords get -100. Best epoch selected by dev token
accuracy (stable on this corpus size; all metrics reported at the end).

Output layout:
    results/checkpoints/{model}_{seed}.pt
    results/dumps/{model}_{seed}_{dev|test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .data import ID2LABEL, LABEL2ID, UPOS_LABELS, load_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_iter(dataset, batch_size: int, shuffle: bool, generator=None):
    idx = torch.randperm(len(dataset), generator=generator) if shuffle \
        else torch.arange(len(dataset))
    for start in range(0, len(dataset), batch_size):
        yield [dataset[i] for i in idx[start:start + batch_size]]


def collate(batch, pad_id: int):
    maxlen = max(len(b["labels"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        pad = maxlen - len(b["labels"])
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        attn.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return torch.stack(input_ids), torch.stack(attn), torch.stack(labels)


@torch.no_grad()
def collect_logits(model, sentences, tokenizer, device, batch_size: int = 32):
    """Per-token (logits, gold_id) with first-subword pooling, in corpus order."""
    model.eval()
    rows = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]
        enc = tokenizer([s.tokens for s in chunk],
                        is_split_into_words=True, truncation=True,
                        max_length=128, padding=True, return_tensors="pt")
        word_ids_per_sample = [enc.word_ids(batch_index=i) for i in range(len(chunk))]
        for k in enc:
            enc[k] = enc[k].to(device)
        logits = model(**enc).logits.float().cpu()
        for i, s in enumerate(chunk):
            seen = set()
            for pos, wid in enumerate(word_ids_per_sample[i]):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                rows.append({
                    "sent_id": s.sent_id,
                    "tok_idx": wid,
                    "token": s.tokens[wid],
                    "gold": s.upos[wid],
                    "gold_id": LABEL2ID[s.upos[wid]],
                    "logits": logits[i, pos].numpy().astype(np.float32),
                })
    return rows


def dev_accuracy(model, dev_sents, tokenizer, device) -> float:
    rows = collect_logits(model, dev_sents, tokenizer, device)
    correct = sum(r["logits"].argmax() == r["gold_id"] for r in rows)
    return correct / len(rows)


def train_one(config_path: str, seed: int, out_root: str,
              max_epochs: int | None = None, patience: int | None = None,
              verbose: bool = True):
    import yaml
    cfg = yaml.safe_load(open(config_path))
    name = cfg["name"]
    max_epochs = max_epochs or cfg.get("max_epochs", 40)
    patience = patience or cfg.get("patience", 5)
    lr = float(cfg.get("lr", 5e-5))
    batch_size = cfg.get("batch_size", 16)

    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    data = load_split(Path(__file__).resolve().parent.parent / "data")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"], num_labels=len(UPOS_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_sents = data["train"]
    dev_sents = data["dev"]
    test_sents = data["test"]

    # Pre-encode training set once.
    train_set = []
    for s in train_sents:
        enc = tokenizer(s.tokens, is_split_into_words=True, truncation=True,
                        max_length=128, return_tensors="pt")
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(LABEL2ID[s.upos[wid]])
            else:
                labels.append(-100)
            prev = wid
        train_set.append({
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    best_acc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in batch_iter(train_set, batch_size, shuffle=True, generator=gen):
            input_ids, attn, labels = collate(batch, tokenizer.pad_token_id)
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += out.loss.item()
            n_batches += 1
        acc = dev_accuracy(model, dev_sents, tokenizer, device)
        if verbose:
            print(f"[{name} seed={seed}] epoch {epoch:3d} "
                  f"loss={total_loss / max(n_batches,1):.4f} dev_acc={acc:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"[{name} seed={seed}] early stop at epoch {epoch} "
                      f"(best {best_epoch}, dev_acc {best_acc:.4f})", flush=True)
            break

    model.load_state_dict(best_state)

    # Save checkpoint + per-token logits for dev and test.
    ckpt_dir = Path(out_root) / "checkpoints"
    dump_dir = Path(out_root) / "dumps"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "config": cfg,
        "seed": seed,
        "best_epoch": best_epoch,
        "dev_accuracy": best_acc,
        "label_map": ID2LABEL,
    }, ckpt_dir / f"{name}_{seed}.pt")

    summary = {"name": name, "seed": seed, "best_epoch": best_epoch,
               "dev_accuracy": best_acc, "train_seconds": time.time() - t0,
               "n_train": len(train_set), "hparams": {"lr": lr,
               "batch_size": batch_size, "max_epochs": max_epochs,
               "patience": patience}}
    for split, sents in (("dev", dev_sents), ("test", test_sents)):
        rows = collect_logits(model, sents, tokenizer, device)
        path = dump_dir / f"{name}_{seed}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "sent_id": r["sent_id"], "tok_idx": r["tok_idx"],
                    "token": r["token"], "gold": r["gold"],
                    "gold_id": int(r["gold_id"]),
                    "logits": [round(float(x), 6) for x in r["logits"]],
                }, ensure_ascii=False) + "\n")
        summary[f"n_{split}"] = len(rows)
    if verbose:
        print(f"[{name} seed={seed}] done: {json.dumps(summary)}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="results")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    args = p.parse_args()
    train_one(args.config, args.seed, args.out_root,
              max_epochs=args.max_epochs, patience=args.patience)


if __name__ == "__main__":
    main()
