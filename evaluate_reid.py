# ewaluacja ReID dla SoccerNet
# działa z checkpointem z treningu

import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# importuj z pliku uzytego do trenowania
from train_soccernet_reid_full import (
    ReIDLit, CFG, parse_filename
)

# czysty transform do ewaluacji
def build_eval_transform():
    H, W = CFG.image_size
    return transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

class ReIDSplit(Dataset):
    def __init__(self, root: str, split: str = "valid"):
        self.root = Path(root)
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Brak katalogu: {split_dir}")
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths += list(split_dir.rglob(ext))
        items = []
        for p in paths:
            meta = parse_filename(p.stem)
            if meta is None:
                continue
            key = f"{meta['action_idx']}|{meta['person_uid']}"
            items.append((str(p), key, int(meta["action_idx"])))

        lab2int = {}
        mapped = []
        for path, key, act in items:
            if key not in lab2int:
                lab2int[key] = len(lab2int)
            mapped.append((path, lab2int[key], act))

        self.items = mapped
        self.labels = [y for _, y, _ in mapped]
        self.actions = [a for *_, a in mapped]
        self.tfm = build_eval_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y, a = self.items[idx]
        img = Image.open(p).convert("RGB")
        img = self.tfm(img)
        return img, y, a, p

def cmc_map_for_action(emb: torch.Tensor, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()
    N = emb.size(0)
    cmc_counts = np.zeros(N, dtype=np.int64)
    ap_list = []
    for i in range(N):
        s = sim[i].clone()
        s[i] = -1e9
        order = torch.argsort(s, descending=True).cpu().numpy()
        rel = (labels[order] == labels[i]).astype(np.int64)
        if rel.sum() == 0:
            continue
        first_hit = np.argmax(rel)
        cmc_counts[first_hit:] += 1
        cumsum = rel.cumsum()
        ranks = np.where(rel == 1)[0]
        prec_at_k = cumsum[ranks] / (ranks + 1)
        ap_list.append(prec_at_k.mean())
    cmc = cmc_counts / max(1, labels.shape[0])
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    return cmc, mAP

def evaluate(split_ds: ReIDSplit, model: ReIDLit, bs: int = 128, num_workers: int = 0):
    loader = DataLoader(
        split_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"      # GPU Apple
    else:
        device = "cpu"
    model.eval().to(device)

    all_emb, all_lab, all_act = [], [], []
    with torch.no_grad():
        for x, y, act, _ in loader:
            x = x.to(device, non_blocking=True)
            e = model(x)
            all_emb.append(e.cpu())
            all_lab.append(y.numpy())
            all_act.append(act.numpy())

    emb = torch.cat(all_emb, dim=0)
    labels = np.concatenate(all_lab, axis=0)
    actions = np.concatenate(all_act, axis=0)

    uniq = np.unique(actions)
    cmc_sum, ap_sum, valid = None, 0.0, 0
    for a in uniq:
        idx = np.where(actions == a)[0]
        if len(idx) < 2:
            continue
        cmc, ap = cmc_map_for_action(emb[idx], labels[idx])
        cmc_sum = cmc if cmc_sum is None else cmc_sum[:len(cmc)] + cmc[:len(cmc_sum)]
        ap_sum += ap
        valid += 1

    if valid == 0:
        raise RuntimeError("Brak akcji do ewaluacji")

    cmc_avg = cmc_sum / valid
    return {
        "rank1": float(cmc_avg[0]),
        "rank5": float(cmc_avg[min(4, len(cmc_avg)-1)]),
        "mAP": float(ap_sum / valid),
    }

def find_last_ckpt(ckpt_dir="checkpoints"):
    # najnowszy po czasie modyfikacji
    paths = sorted(Path(ckpt_dir).glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError("Nie znaleziono checkpointów")
    return str(paths[-1])

def main():
    split = "valid"
    ckpt = find_last_ckpt()
    print(f"Używam split: {split}")
    print(f"Używam ckpt:  {ckpt}")

    ds = ReIDSplit(CFG.data_root, split=split)

    model = ReIDLit.load_from_checkpoint(ckpt, cfg=CFG, strict=False)

    nw = 0

    metrics = evaluate(ds, model, bs=128, num_workers=nw)
    print("rank-1:", metrics["rank1"])
    print("rank-5:", metrics["rank5"])
    print("mAP:", metrics["mAP"])

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "results_valid.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    print("Zapisano do results/results_valid.txt")

if __name__ == "__main__":
    main()
