# ewaluacja ReID dla SoccerNet v3 (JSON)
# wykorzystuje checkpoint z treningu

import json
import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from collections import Counter


# import z pliku treningowego
from train_soccernet_reid_full import (
    ReIDLit,
    CFG,
)

# ===================== TRANSFORM =====================

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

# ===================== DATASET (JSON, QUERY/GALLERY) =====================

class ReIDJSONSplit(Dataset):
    """
    Dataset do ewaluacji
    Korzysta z bbox_info.json i podziału na
    query / gallery w katalogach valid / test

    label = action_idx | person_uid
    """
    def __init__(self, root: str,
                 split: str = "valid",
                 group: str = "query"):
        """
        root  -> np "dataSoccerNet/reid-2023"
        split -> "valid" lub "test"
        group -> "query" lub "gallery"
        """
        self.root = Path(root)
        self.split = split
        self.group = group

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Brak katalogu split: {split_dir}")

        # ścieżka do JSON
        if split == "train":
            json_name = "train_bbox_info.json"
        else:
            json_name = "bbox_info.json"

        json_path = split_dir / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"Brak pliku JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta_root = json.load(f)

        # train ma płaski dict
        # valid/test mają {"query": {...}, "gallery": {...}}
        if split in ("valid", "test"):
            if group not in meta_root:
                raise KeyError(f"Brak grupy {group} w {json_path}")
            meta_all = meta_root[group]
            base_dir = split_dir / group
        else:
            meta_all = meta_root
            base_dir = split_dir

        # klasy jak w treningu
        allowed_classes = {
            "Player_team_left",
            "Player_team_right",
            "Goalkeeper_team_left",
            "Goalkeeper_team_right",
            "Main_referee",
            "Side_referee",
            "Goalkeeper_team_unknown",
            "Player_team_unknown_1",
            "Player_team_unknown_2",
        }

        samples = []
        missing = 0
        total = 0

        for _, info in meta_all.items():
            clazz = info["clazz"]
            if clazz not in allowed_classes:
                continue

            bbox_idx = info["bbox_idx"]
            action_idx = info["action_idx"]
            person_uid = info["person_uid"]
            frame_idx = info["frame_idx"]
            rel_path = info["relative_path"]
            pid_in_action = info["id"]
            uai = info["UAI"]
            h = info["height"]
            w = info["width"]

            pid_str = str(pid_in_action)

            file_name = (
                f"{bbox_idx}-"
                f"{action_idx}-"
                f"{person_uid}-"
                f"{frame_idx}-"
                f"{clazz}-"
                f"{pid_str}-"
                f"{uai}-"
                f"{h}x{w}.png"
            )

            img_path = base_dir / rel_path / file_name
            total += 1

            if not img_path.exists():
                missing += 1
                continue

            samples.append(
                (str(img_path), int(person_uid), int(action_idx))
            )


        print(
            f"[{split}:{group}] total={total} "
            f"missing={missing} samples={len(samples)}"
        )

        if not samples:
            raise RuntimeError(
                f"Brak próbek w {split}:{group} po filtracji"
            )

        self.items = samples
        self.labels = [y for _, y, _ in samples]
        self.actions = [a for *_, a in samples]
        self.tfm = build_eval_transform()

        # ===== DEBUG rozkładu ID =====
        cnt = Counter(self.labels)
        vals = np.array(list(cnt.values()))
        print(f"[{split}:{group}] N obrazów = {len(self.items)}")
        print(f"[{split}:{group}] unikalnych ID = {len(cnt)}")
        print(f"[{split}:{group}] średnia próbek na ID = {vals.mean():.2f}")
        print(f"[{split}:{group}] min próbek na ID = {vals.min()}")
        print(f"[{split}:{group}] max próbek na ID = {vals.max()}")
        one_sample = (vals == 1).sum()
        print(f"[{split}:{group}] ID z 1 próbką = {one_sample}")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y, act = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return img, y, act, path

# ===================== METRYKI CMC + mAP =====================

def cmc_map_query_gallery(
    q_emb: torch.Tensor,
    q_labels: np.ndarray,
    g_emb: torch.Tensor,
    g_labels: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Standardowe ReID
    szukamy w gallery dla każdego query
    """

    # upewniamy się że embeddingi są L2
    q_emb = F.normalize(q_emb, p=2, dim=1)
    g_emb = F.normalize(g_emb, p=2, dim=1)

    sim = q_emb @ g_emb.t()   # cosinus
    num_q, num_g = sim.shape

    cmc_counts = np.zeros(num_g, dtype=np.int64)
    ap_list = []

    for i in range(num_q):
        s = sim[i]

        # sortujemy gallery po podobieństwie
        order = torch.argsort(s, descending=True).cpu().numpy()

        rel = (g_labels[order] == q_labels[i]).astype(np.int64)
        if rel.sum() == 0:
            # brak pozytywnych w gallery
            continue

        # CMC
        first_hit = np.argmax(rel)
        cmc_counts[first_hit:] += 1

        # AP
        cumsum = rel.cumsum()
        pos_idx = np.where(rel == 1)[0]
        prec_at_k = cumsum[pos_idx] / (pos_idx + 1)
        ap_list.append(prec_at_k.mean())

    cmc = cmc_counts / max(1, num_q)
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    return cmc, mAP

# ===================== EKSTRAKCJA EMBEDDINGÓW =====================

def extract_embeddings(
    ds: Dataset,
    model: ReIDLit,
    device: str,
    bs: int = 128,
    num_workers: int = 0,
):
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_emb, all_lab, all_act = [], [], []

    model.eval().to(device)

    with torch.no_grad():
        for x, y, act, _ in loader:
            x = x.to(device, non_blocking=True)
            e = model(x)  # ReIDLit.forward -> embedding
            all_emb.append(e.cpu())
            all_lab.append(y.numpy())
            all_act.append(act.numpy())

    emb = torch.cat(all_emb, dim=0)
    labels = np.concatenate(all_lab, axis=0)
    actions = np.concatenate(all_act, axis=0)
    return emb, labels, actions

# ===================== GŁÓWNA FUNKCJA EWALUACJI =====================

def evaluate(
    root: str,
    model: ReIDLit,
    split: str = "valid",
    bs: int = 128,
    num_workers: int = 0,
):
    # wybór urządzenia
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # query i gallery jako osobne datasety
    ds_query = ReIDJSONSplit(root, split=split, group="query")
    ds_gallery = ReIDJSONSplit(root, split=split, group="gallery")

    q_emb, q_labels, _ = extract_embeddings(
        ds_query, model, device, bs=bs, num_workers=num_workers
    )
    g_emb, g_labels, _ = extract_embeddings(
        ds_gallery, model, device, bs=bs, num_workers=num_workers
    )

    cmc, mAP = cmc_map_query_gallery(
        q_emb, q_labels, g_emb, g_labels
    )

    rank1 = float(cmc[0])
    rank5 = float(cmc[min(4, len(cmc) - 1)])

    return {
        "rank1": rank1,
        "rank5": rank5,
        "mAP": float(mAP),
    }

# ===================== CHECKPOINT =====================

def find_last_ckpt(ckpt_dir="checkpoints"):
    paths = sorted(
        Path(ckpt_dir).glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime
    )
    if not paths:
        raise FileNotFoundError("Nie znaleziono checkpointów")
    return str(paths[-1])

# ===================== MAIN =====================

def main():
    split = "valid"  # albo "test"
    ckpt = find_last_ckpt()
    print(f"Używam split: {split}")
    print(f"Używam ckpt:  {ckpt}")

    model = ReIDLit.load_from_checkpoint(
        ckpt, cfg=CFG, strict=False
    )

    nw = 0  # num_workers dla DataLoader

    metrics = evaluate(
        CFG.data_root,
        model,
        split=split,
        bs=128,
        num_workers=nw,
    )

    print("rank-1:", metrics["rank1"])
    print("rank-5:", metrics["rank5"])
    print("mAP:", metrics["mAP"])

    out = Path("results")
    out.mkdir(exist_ok=True)
    out_file = out / f"results_{split}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    print(f"Zapisano do {out_file}")

if __name__ == "__main__":
    main()
