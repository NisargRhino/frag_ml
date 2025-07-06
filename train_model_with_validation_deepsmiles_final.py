from __future__ import annotations
import argparse, csv, json, os, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (RobertaForMaskedLM, RobertaTokenizer, get_linear_schedule_with_warmup)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
RDLogger.DisableLog("rdApp.*")

try:
    from deepsmiles import Converter
except ImportError:
    raise SystemExit("Install deepsmiles first:  pip install deepsmiles")
_conv = Converter(rings=True, branches=True)


def deep_to_smiles(ds: str) -> str | None:
    try:
        return _conv.decode(ds)
    except Exception:
        return None


def tanimoto(sm1: str | None, sm2: str | None) -> float:
    if sm1 is None or sm2 is None:
        return 0.0
    m1, m2 = Chem.MolFromSmiles(sm1), Chem.MolFromSmiles(sm2)
    if m1 is None or m2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def export_full_package(run_dir: Path, model: RobertaForMaskedLM, tokenizer: RobertaTokenizer, meta: dict):
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    with open(run_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)


class SMILESDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: RobertaTokenizer, *, smiles_col="DRUG SMILES_DEEP", label_col="FRAG_SMILES_DEEP",max_length=128):
        df = df.dropna(subset=[smiles_col, label_col])
        df = df[(df[smiles_col].str.len() > 0) & (df[label_col].str.len() > 0)]
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.smiles_col, self.label_col = smiles_col, label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _enc(self, s: str):
        return self.tok(s, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inp = self._enc(row[self.smiles_col])
        lbl = self._enc(row[self.label_col])
        labels = lbl["input_ids"].squeeze()
        labels[lbl["attention_mask"].squeeze() == 0] = -100
        return {"input_ids": inp["input_ids"].squeeze(),
                "attention_mask": inp["attention_mask"].squeeze(),
                "labels": labels,
                "gold_deepsmiles": row[self.label_col]}


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return out


def decode_batch(tok, ids): return [s.replace(" ", "") for s in
                                    tok.batch_decode(ids, skip_special_tokens=True)]


def epoch_loop(model, dl, tok, optim, sched, device, *, is_train, λ_seq, λ_l1, λ_l2, invalid_penalty):
    getattr(model, "train" if is_train else "eval")()
    running_loss = running_tani = n = 0
    pbar = tqdm(dl, desc="Train" if is_train else "Eval ")
    for batch in pbar:
        gold_ds = batch["gold_deepsmiles"]
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.set_grad_enabled(is_train):
            out = model(**batch_t)
            ce = out.loss
            preds_ds = decode_batch(tok, out.logits.argmax(-1))
            preds_sm = [deep_to_smiles(s) for s in preds_ds]
            gold_sm = [deep_to_smiles(s) for s in gold_ds]
            tani = np.fromiter((tanimoto(p, g) for p, g in zip(preds_sm, gold_sm)),dtype=float)
            mean_t = tani.mean() if len(tani) else 0.0
            seq_pen = 1.0 - mean_t
            inv_loss = np.fromiter((p is None for p in preds_sm), float).mean() * invalid_penalty
            l1 = sum(p.abs().sum() for p in model.parameters())
            l2 = sum(p.pow(2).sum() for p in model.parameters())
            loss = ce + λ_seq * seq_pen + inv_loss + λ_l1 * l1 + λ_l2 * l2
            if is_train:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
        bs = batch_t["input_ids"].size(0)
        running_loss += loss.item() * bs
        running_tani += mean_t * bs
        n += bs
        pbar.set_postfix(loss=running_loss / n, tani=running_tani / n)
    return running_loss / n, running_tani / n


def plot_two_series(xs, ys1, ys2, ylabel, out_path, title):
    plt.figure()
    plt.plot(xs, ys1, label="train")
    plt.plot(xs, ys2, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--checkpoint", default="Deepsmiles-chempbl1m-04162025")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_length", type=int, default=45)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--resume")
    args = ap.parse_args()

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        ckpt_path = Path(args.resume).resolve()
        if not ckpt_path.is_file(): raise FileNotFoundError(ckpt_path)
        run_dir = ckpt_path.parent
    else:
        run_dir = Path("runs") / datetime.now().strftime("%y-%m-%d-%H-%M-%S_deepsmiles")
        run_dir.mkdir(parents=True, exist_ok=True)

    tok = RobertaTokenizer.from_pretrained(args.checkpoint)
    model = RobertaForMaskedLM.from_pretrained(args.checkpoint).to(device)

    train_ds = SMILESDataset(pd.read_csv(args.train), tok, max_length=args.max_length)
    val_ds = SMILESDataset(pd.read_csv(args.val), tok, max_length=args.max_length)
    test_ds = SMILESDataset(pd.read_csv(args.test), tok, max_length=args.max_length)

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, 1, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, 0, len(train_dl) * args.epochs)

    best_tani, patience, start_epoch = 0.0, args.patience, 1
    if args.resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"· Resuming from {ckpt_path}")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        best_tani = ckpt["best_tani"]
        start_epoch = ckpt["epoch"] + 1

    λ_seq, λ_l1, λ_l2, invalid_pen = 8, 0, 1e-5, 2.5


    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_tani": [], "val_tani": []}
    ft_hist = {k: [] for k in hist} if args.resume else None

    try:
        for ep in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {ep}/{args.epochs}")
            tr_loss, tr_t = epoch_loop(model, train_dl, tok, optim, sched, device,is_train=True, λ_seq=λ_seq, λ_l1=λ_l1, λ_l2=λ_l2, invalid_penalty=invalid_pen)
            vl_loss, vl_t = epoch_loop(model, val_dl, tok, optim, sched, device,is_train=False, λ_seq=λ_seq, λ_l1=λ_l1, λ_l2=λ_l2, invalid_penalty=invalid_pen)

            print(f"train loss {tr_loss:.4f} | val loss {vl_loss:.4f}")
            print(f"train tani  {tr_t:.3f} | val tani  {vl_t:.3f}")

            hist["epoch"].append(ep)
            hist["train_loss"].append(tr_loss)
            hist["val_loss"].append(vl_loss)
            hist["train_tani"].append(tr_t)
            hist["val_tani"].append(vl_t)
            if ft_hist is not None:
                ft_hist["epoch"].append(ep)
                ft_hist["train_loss"].append(tr_loss)
                ft_hist["val_loss"].append(vl_loss)
                ft_hist["train_tani"].append(tr_t)
                ft_hist["val_tani"].append(vl_t)

            if vl_t > best_tani + 1e-4:
                best_tani = vl_t
                patience = args.patience
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(),"sched": sched.state_dict(), "epoch": ep, "best_tani": best_tani}, run_dir / "best_model.pt")
                export_full_package(run_dir, model, tok,
                                    {"epoch": ep, "best_val_tanimoto": best_tani, "args": vars(args),
                                     "timestamp": datetime.now().isoformat(timespec="seconds")})
                print("  ↳ saved best_model.pt + full package")
            else:
                patience -= 1
                if patience == 0: print("Early stopping"); break
    except KeyboardInterrupt:
        torch.save({"model": model.state_dict()}, run_dir / "last_interrupt.pt")
        print("\n⇢ training interrupted – saved last_interrupt.pt")
        return


    plot_two_series(hist["epoch"], hist["train_tani"], hist["val_tani"], "Mean Tanimoto", run_dir / "tanimoto_curve.png", "Tanimoto similarity vs epoch")
    plot_two_series(hist["epoch"], hist["train_loss"], hist["val_loss"], "Loss", run_dir / "loss_curve.png", "Loss vs epoch")
    if ft_hist and ft_hist["epoch"]:
        plot_two_series(ft_hist["epoch"], ft_hist["train_tani"], ft_hist["val_tani"], "Mean Tanimoto", run_dir / "finetune_tanimoto_curve.png", "Tanimoto similarity (fine-tune)")
        plot_two_series(ft_hist["epoch"], ft_hist["train_loss"], ft_hist["val_loss"], "Loss", run_dir / "finetune_loss_curve.png", "Loss (fine-tune)")


    print("\nEvaluating best model on test set…")
    best = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model"])
    ts_loss, ts_t = epoch_loop(model, test_dl, tok, optim, sched, device,
                               is_train=False, λ_seq=λ_seq, λ_l1=λ_l1,
                               λ_l2=λ_l2, invalid_penalty=invalid_pen)
    print(f"Test Tanimoto {ts_t:.3f}")
    print(f"Run finished in {(time.time() - start) / 60:.1f} min")

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            gold_ds = batch["gold_deepsmiles"][0]
            gold_sm = deep_to_smiles(gold_ds)
            batch_cuda = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred_ids = model(**batch_cuda).logits.argmax(-1)
            pred_ds = decode_batch(tok, pred_ids)[0]
            pred_sm = deep_to_smiles(pred_ds)
            rows.append({"drug_deep": tok.batch_decode(batch_cuda["input_ids"], skip_special_tokens=True)[0].replace(" ", ""),
                         "gold_deep": gold_ds, "pred_deep": pred_ds,
                         "pred_valid": pred_sm is not None,
                         "tanimoto": tanimoto(pred_sm, gold_sm)})
    csv_path = run_dir / "test_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ full test predictions written to {csv_path}")


if __name__ == "__main__":
    main()
