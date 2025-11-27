import os
import re
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import json

import timm
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# straty i minery
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.losses import (
    TripletMarginLoss,
    MultiSimilarityLoss,
)
from pytorch_metric_learning.miners import (
    BatchHardMiner,
    MultiSimilarityMiner,
)
from pytorch_metric_learning.distances import CosineSimilarity


# ===================== CONFIG =====================
class CFG:
    # ścieżka do danych
    data_root = "dataSoccerNet/reid-2023"

    # rozmiar wejścia HxW
    image_size = (288, 144)

    # batch i workers
    batch_size = 96
    num_workers = 4

    # epoki i lr
    epochs = 20
    lr = 3e-4  # Startowy lr dla AdamW

    # embedding i arch
    embedding_dim = 512  # Wymiar wektora cech po head
    arch = "resnet50d"  # Backbone z timm. 50d ma lepszy stem

    # PK sampler
    K = 4  # liczba próbek na klasę w PK Samplerze. Batch to P x K przykładów.

    # precyzja
    precision = "16-mixed"

    # seed i freeze
    seed = 42  # Deterministyczne losowanie
    freeze_backbone_epochs = 1  # warm start -> Backbone zamrażany na 1 epokę. Uczy się najpierw tylko głowa -> Stabilizacja embeddingów na starcie

    # loss i miner
    loss_name = "ms"  # ms lub triplet -> Strata metryczna
    use_miner = True  # Włączenie doboru tródnych par
    miner_warmup_epochs = 2  # opóźnij miner o 2 epoki
    
    ''' 
    Gdy OOM zmniejsz batch lub HxW
    Gdy niestabilnie zmniejsz LR
    Gdy brak poprawy zwiększ epoki
    Gdy overfit dodaj erasing lub cutout
    Gdy mało pozytywów podnieś K
    Gdy PK się sypie zmniejsz K lub batch
    '''


# ===================== UTILS =====================
'''
Ustawienie ziarna dla wszystkich bibliotek
'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

'''
Zebranie wszystkich plików graficznych z katalogu
'''
def list_images(split_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files += list(split_dir.rglob(ext))
    return files


'''
Parser nazwy pliku. Wyrażenie regularne REGEX

Przykładowa nazwa: 0123-45-6789-100-class-ID-UAI-256x128.png

bbox_idx	indeks wycinka bounding boxa
action_idx	numer akcji (np. konkretne ujęcie meczu)
person_uid	unikalny identyfikator zawodnika
frame_idx	numer klatki w tej akcji
class	klasa rozpoznawania (np. player, referee, ball)
ID	wewnętrzny ID w datasetcie
UAI	identyfikator dodatkowy (np. identyfikator ujęcia akcji)
h i w	wysokość i szerokość obrazka w pikselach
'''
_RX = re.compile(
    r"^(?P<bbox_idx>\d+)-"
    r"(?P<action_idx>\d+)-"
    r"(?P<person_uid>\d+)-"
    r"(?P<frame_idx>\d+)-"
    r"(?P<class>[^-]+)-"
    r"(?P<ID>[^-]+)-"
    r"(?P<UAI>[^-]+)-"
    r"(?P<h>\d+)x(?P<w>\d+)$"
)

'''
Funkcja parsująca - dla podanej nazwy bez rozszerzenia zwraca słownik
metadanych
Przykład:
{
  "bbox_idx": "0123",
  "action_idx": "45",
  "person_uid": "6789",
  "frame_idx": "100",
  "class": "player",
  "ID": "xyz",
  "UAI": "abc",
  "height": 256,
  "width": 128
}
'''
def parse_filename(stem: str):
    m = _RX.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["height"] = int(d.pop("h"))
    d["width"] = int(d.pop("w"))
    return d


# ===================== DATASET =====================
'''
Funkcja przygotowująca łańcuch transformacji obrazu dla zbioru 
treningowego oraz walidacyjnego
'''
def build_transform(train=True):
    H, W = CFG.image_size
    '''
    Trening - nauka różnorodności wyglądu tego samego gracza. Lekkie augmentacje - lista kroków wykonywana po kolei
        - Resize: Zmiana rozmiaru obrazu - wszystkie obrazy w batchu muszą mieć ten sam wymiar
        - RandomHorizontalFlip: Odbicie lustrzane z prawdopodobieństwem 50%. Pomaga modelowi zrozumieć, że gracz w odbiciu to ta sama osoba. 
        Kamery mogą być po lewej/prawej stronie boiska
        - RandomApply: Z prawdopodobieństwem 80% zastosuje jedną z podanych augemntacji. W tym przypadku ColorJitter, jasność ±20%, kontrast ±20%, nasycenie ±20%, odcień ±0.02
        - RandomGrayscale: Zmienia obraz na czarno-biały z szansą 5%. Zmusza model, by patrzył też na kształt sylwetki, nie tylko kolor
        - ToTensor: Zamiana obrazu PIL (0-255) na tensor PyTorch (0-1). HxWxC -> CxHxW
        - Normalize: Normalizacja kanałów RGB tak jak w ImageNet. Odejmuje średnią i dzieli przez odchylenie standardowe.
        Średnie kolory kanałów:
        R: 0.485
        G: 0.456
        B: 0.406

        Odchylenia standardowe:
        R: 0.229
        G: 0.224
        B: 0.225

        Te wartości są stałe, bo backbone (ResNet50d) był trenowany na ImageNet z takim samym preprocessingiem. Dzięki temu dane mają podobny rozkład jak te, na których sieć była wstępnie nauczona
        - RandomErasing: Z prawdopodobieństwem 60% usuwa losowy prostokąt z obrazu (zamienia go na losowy kolor). Symuluje zasłonięcia (np. inny zawodnik, ręka, cień)
        scale=(0.02, 0.2) — wymazany obszar stanowi 2-20% powierzchni obrazu

    '''
    if train:
        return transforms.Compose([
            transforms.Resize((H, W)),  
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.02)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(
                p=0.6, scale=(0.02, 0.2), value="random"
            ),
        ])
    # Tryb walidacji/testu - czyste, powtrarzalne wejście
    else:
        return transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

'''
Klasa przygotowująca dane dla sieci
Wersja z wykorzystaniem plików JSON
train_bbox_info.json / valid_bbox_info.json

Kroki
    - wczytaj metadane z JSON
    - zbuduj ścieżkę do obrazka z pól JSON
    - odfiltruj mniej ważne klasy (np Staff)
    - nadaj etykiety ID w obrębie całego zbioru
    - zbuduj mapę action_idx -> lista indeksów
      do późniejszego samplingu per-akcja
'''
class SoccerNetReID(Dataset):
    # dataset do ReID
    # label = action_idx + person_uid
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split  # Podfolder
        split_dir = self.root / split   # np dataSoccerNet/reid-2023/train
        if not split_dir.exists():
            raise FileNotFoundError(f"brak katalogu: {split_dir}")

        '''
        Plik z metadanymi bbox z JSON
            train  -> train/train_bbox_info.json
            valid  -> valid/bbox_info.json
            test   -> test/bbox_info.json
            challenge na razie bez JSON
        '''
        json_dir = self.root / split

        if split == "train":
            json_name = "train_bbox_info.json"
        elif split == "challenge":
            raise NotImplementedError("challenge bez JSON")
        else:
            json_name = "bbox_info.json"

        json_path = json_dir / json_name

        if not json_path.exists():
            raise FileNotFoundError(f"brak pliku JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta_root = json.load(f)

        '''
        Rozpakowanie struktur JSON
            train  -> jeden słownik wpisów
            valid  -> dwa słowniki: query i gallery
            test   -> dwa słowniki: query i gallery

        groups to lista krotek
            (nazwa_grupy, słownik_z_wpisami)

        nazwa_grupy jest
            ""        dla train
            "query"   dla valid/test query
            "gallery" dla valid/test gallery
        '''
        groups = []
        if split in ("valid", "test"):
            if "query" in meta_root:
                groups.append(("query", meta_root["query"]))
            if "gallery" in meta_root:
                groups.append(("gallery", meta_root["gallery"]))
        else:
            # train ma wszystko w jednym dict
            groups.append(("", meta_root))

        print(
            f"[{split}] grupy w JSON:",
            [g[0] or "root" for g in groups]
        )

        '''
        Lista do przechowania próbek
        Każdy element:
            (pełna_ścieżka_png, klucz_ID, action_idx, klasa_tekstowa)
        '''
        samples = []

        '''
        Do balansowania klas semantycznych
        Bierzemy tylko graczy i sędziów
        Pomijamy np Staff itp
        '''
        allowed_classes = {
            "Player_team_left",
            "Player_team_right",
            "Goalkeeper_team_left",
            "Goalkeeper_team_right",
            "Main_referee",
            "Side_referee",
            "Goalkeeper_team_unknown",
            "Goalkeeper_team_left_unknown",
            "Goalkeeper_team_right_unknown",
            "Player_team_unknown_1",
            "Player_team_unknown_2",
        }

        from collections import Counter
        all_classes = Counter()

        missing = 0   # liczba brakujących plików
        total = 0     # liczba prób po filtrach klas

        '''
        Iterujemy po grupach
        train  -> jedna grupa ""
        valid  -> dwie grupy "query" i "gallery"
        test   -> dwie grupy "query" i "gallery"
        '''
        for group_name, meta_all in groups:
            '''
            Bazowy katalog dla tej grupy
                train  -> .../train
                valid  -> .../valid/query lub .../valid/gallery
                test   -> .../test/query lub .../test/gallery
            '''
            base_dir = (
                split_dir / group_name if group_name else split_dir
            )

            # aktualizujemy licznik klas
            all_classes.update(
                info["clazz"] for info in meta_all.values()
            )

            for _, info in meta_all.items():
                clazz = info["clazz"]

                # filtr klas semantycznych
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

                # id może być None lub literą
                pid_str = str(pid_in_action)

                '''
                Nazwa pliku zgodna ze specyfikacją
                <bbox_idx>-<action_idx>-<person_uid>-<frame_idx>
                -<clazz>-<ID>-<UAI>-<height>x<width>.png
                '''
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
                    # podgląd pierwszych braków w valid
                    if missing < 10 and split == "valid":
                        print("Brak pliku:", img_path)
                    missing += 1
                    continue

                '''
                Klucz ID
                Tożsamość ważna tylko w obrębie jednej akcji
                Dlatego składamy:
                    action_idx | person_uid
                '''
                key = f"{action_idx}|{person_uid}"
                samples.append(
                    (str(img_path), key, int(action_idx), clazz)
                )

        print(f"[{split}] klasy w JSON:", all_classes)
        print(f"[{split}] total={total} missing={missing}")
        print(f"[{split}] samples={len(samples)}")

        if not samples:
            raise RuntimeError(
                f"Brak próbek po filtracji w {split}"
            )

        '''
        Remap klucza ID na int
            "12|3456" -> 0
            "12|7890" -> 1
            itd
        Wynik końcowy:
            items = (path, label_int, action_idx, clazz)
        '''
        label_to_int = {}
        items = []
        for path, key, act, clazz in samples:
            if key not in label_to_int:
                label_to_int[key] = len(label_to_int)
            items.append((path, label_to_int[key], act, clazz))

        # zapis pól do klasy
        self.items = items  # główna lista z danymi
        self.labels = [y for _, y, _, _ in items]   # etykiety ID
        self.actions = [a for _, _, a, _ in items]  # indeksy akcji
        self.classes = [c for *_, c in items]       # klasy semantyczne

        '''
        Mapa akcji do indeksów datasetu
        action_idx -> [i0, i1, i2, ...]
        Przydatne do samplingu per-akcja
        '''
        self.action_to_indices = {}
        for idx, (_, _, act, _) in enumerate(items):
            self.action_to_indices.setdefault(act, []).append(idx)

        # transformacje
        if transform is None:
            transform = build_transform(train=(split == "train"))
        self.transform = transform

    '''
    Zwraca liczbę wszystkich próbek
    by DataLoader widział ile kroków ma epoka
    '''
    def __len__(self):
        return len(self.items)

    '''
    - wybiera rekord po indeksie
    - otwiera obraz z dysku przez PIL.Image.open
    - konwertuje do RGB
    - stosuje transformacje
    Zwraca:
        img  tensor 3xHxW
        y    etykieta (int, ID gracza)
        act  numer akcji
        path ścieżka
    '''
    def __getitem__(self, idx):
        path, y, act, clazz = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y, act, path
    
'''
Sampler akcyjny z PK w obrębie jednej akcji

Założenia
    - każdy batch pochodzi z jednej akcji
    - w batchu jest P identytetów (graczy)
    - dla każdej tożsamości losujemy K przykładów
    - batch_size = P x K

Dzięki temu
    - pozytywy i negatywy pochodzą z tej samej akcji
    - model uczy się trudnych przypadków replayów
'''
class ActionPKSampler(Sampler):
    def __init__(self, labels, actions, action_to_indices,
                 batch_size, K):
        super().__init__()
        self.labels = np.asarray(labels, dtype=np.int64)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.action_to_indices = action_to_indices
        self.batch_size = batch_size
        self.K = K
        self.P = batch_size // K

        '''
        Dla każdej akcji budujemy mapę
        label_int -> lista indeksów w tej akcji
        '''
        self.action_label_to_indices = {}
        for act, idxs in action_to_indices.items():
            lab2idx = {}
            for i in idxs:
                y = int(self.labels[i])
                lab2idx.setdefault(y, []).append(i)
            self.action_label_to_indices[act] = lab2idx

        '''
        Lista akcji które mają sens
        co najmniej 2 różne ID w akcji
        inaczej ReID byłoby zbyt słabe
        '''
        self.valid_actions = [
            act for act, lab2idx
            in self.action_label_to_indices.items()
            if len(lab2idx) >= 2
        ]

        if not self.valid_actions:
            raise RuntimeError("Brak akcji z >=2 ID")

        # przybliżona liczba batchy na epokę
        self.num_batches = len(self.labels) // self.batch_size

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        result_indices = []

        for _ in range(self.num_batches):
            # wybierz losową akcję
            act = int(rng.choice(self.valid_actions))
            lab2idx = self.action_label_to_indices[act]
            labels_in_action = list(lab2idx.keys())

            # wybierz P tożsamości w tej akcji
            if len(labels_in_action) >= self.P:
                chosen_labels = rng.choice(
                    labels_in_action,
                    size=self.P,
                    replace=False,
                )
            else:
                # jak za mało ID, losujemy z powtórzeniami
                chosen_labels = rng.choice(
                    labels_in_action,
                    size=self.P,
                    replace=True,
                )

            # dla każdej tożsamości losujemy K przykładów
            for y in chosen_labels:
                idxs = lab2idx[int(y)]
                if len(idxs) >= self.K:
                    chosen = rng.choice(
                        idxs, size=self.K, replace=False
                    )
                else:
                    chosen = rng.choice(
                        idxs, size=self.K, replace=True
                    )
                result_indices.extend(chosen.tolist())

        return iter(result_indices)

# ===================== MODEL =====================
'''
Klasa łącząca fetaure extractor (backbone) i embedding head (BNNeck). Zwraca gotowy embedding L2.
przyjęcie obrazu gracza -> wyciągnięcie cech wizualnych (backbone) 
-> przekształcenie ich w wektor embeddingu (512D) (Head)

Obraz -> CNN -> avg pool -> 2048D
-> Linear -> BN -> 512D -> L2
-> embedding gotowy do cosinusów
'''
class ReIDBackbone(nn.Module):
    # backbone + głowa do embeddingu o wymiarze 512
    def __init__(self, arch="resnet50d", embed_dim=512):
        super().__init__()
        # tworzenie backbone
        self.backbone = timm.create_model(
            arch, pretrained=True, num_classes=0, global_pool="avg" # avg pooling -> liczenie średniej na każdym kanale
        )
        feat_dim = self.backbone.num_features   # rozmiar wektora po poolingu (2048)

        # head BNneck
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embed_dim, bias=False), # warstwa liniowa: rzutowanie 2048 -> 512
            nn.BatchNorm1d(embed_dim),  # normalizacja wsadowa - stabilizacja rozkładu
        )

        # inicjalizacja wag Linear i BN
        nn.init.normal_(self.head[0].weight, std=0.001)
        nn.init.constant_(self.head[1].weight, 1.0)
        nn.init.constant_(self.head[1].bias, 0.0)

    def forward(self, x):
        '''
        - x to obraz 3xHxW po transformach
        - f to wektor cech z backbone
        - z to rzut na wymiar embed_dim
        - normalize robi L2 po wymiarach
        - zwracamy embedding o normie 1
        '''
        f = self.backbone(x)        # cechy z CNN
        z = self.head(f)            # rzut do embed
        z = F.normalize(z, p=2, dim=1)  # L2 norm
        return z    # embedding o normie 1

'''
fabryka zwracająca 2 elementy:
    - loss_fn - funkcja straty metrycznej (czego model ma się nauczyć)
    - miner - narzędzie do wybierania najtrudniejszych przykładów z batcha
'''
def build_loss(name="ms", use_miner=True):
    # wybór rodzaju straty i minera
    cos = CosineSimilarity()
    if name == "ms":
        '''
        MultiSimilarity - wybieramy wiele par z całego batcha na raz. Porównanie każdy z każdym.
        '''
        loss_fn = MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        miner = MultiSimilarityMiner(epsilon=0.1) if use_miner else None    # Miner szuka tylko trudnych przykładów, gdzie odległość równa ok. granicy decyzji
    else:
        '''
        Triplet jako baza - wybieramy tylko jedną trójkę (anchor, positive, negative).
        '''
        loss_fn = TripletMarginLoss(margin=0.3, distance=cos)
        miner = BatchHardMiner(distance=cos) if use_miner else None # twardy miner: wybiera w batchu najtrudniejsze przykłady: najbardziej odległy pos i najbliższy neg
    return loss_fn, miner

'''
Klasa łącząca model, stratę, opt i logikę treningu
'''
class ReIDLit(pl.LightningModule):
    # Lightning module z treningiem
    def __init__(self, cfg):
        super().__init__()

        # Zmiana klasy CFG na zwykły słownik - zapis hparams bez metod
        def _to_dict(c):
            return {
                k: getattr(c, k)
                for k in dir(c)
                if not k.startswith("_")
                and not callable(getattr(c, k))
            }

        # trwały zapis konfiguracji w modelu
        self.save_hyperparameters(_to_dict(cfg))

        # sieć i strata
        self.model = ReIDBackbone(cfg.arch, cfg.embedding_dim)
        self.loss_fn, self.miner = build_loss(
            cfg.loss_name, cfg.use_miner
        )

        # flagi stanu
        self.frozen = False # mówi czy backbone jest obecnie zamrożony, czli jego parametry nie są uczone
        self.miner_warmup = cfg.miner_warmup_epochs # mówi od której epoki włączyć minera

    # Przepuszczenie btacha obrazów (x) przez model
    def forward(self, x):
        # zwraca embedding L2
        return self.model(x)

    # Freezing i unfreezing backbone’a
    def on_train_epoch_start(self):
        '''
        Zamrożenie backbone na start
            - przechodzi po wszystkich parametrach backbone'a (for p in ...)
            - ustawia requires_grad = False — wyłącza obliczanie gradientu
            - czyli backbone nie uczy się, jego wagi są stałe
            - uczy się tylko head (Linear + BatchNorm)
            - na końcu ustawia flagę self.frozen = True, żeby pamiętać, że już zamroziliśmy
        '''
        if self.current_epoch < CFG.freeze_backbone_epochs and not self.frozen:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
            self.frozen = True

        '''
        Odmrożenie backbone po warm start
            - przechodzi po wszystkich parametrach backbone'a
            - ustawia requires_grad = True — włącza obliczanie gradientu
            - od tego momentu cały model uczy się razem
            - flaga self.frozen zmienia się na False
        '''
        if self.current_epoch >= CFG.freeze_backbone_epochs and self.frozen:
            for p in self.model.backbone.parameters():
                p.requires_grad = True
            self.frozen = False

    '''
    Iteracja treningu
        - zaciągnięcie batcha danych
        - obliczanie embeddingów
        - wybór trudnych przykładów (jeśli miner aktywny)
        - obliczanie straty
        - zwrócenie straty do optymalizacji
    '''
    def training_step(self, batch, batch_idx):
        '''
        Batch danych
            | Zmienna | Znaczenie              | Typ               |
            | ------- | ---------------------- | ----------------- |
            | x     | batch obrazów          | tensor [B, 3, H, W] |
            | y     | etykiety ID zawodników | tensor [B]          |
            | act   | indeksy akcji          | tensor [B]          |
            | path  | ścieżki do plików      | lista stringów      |

        '''
        x, y, act, path = batch
        y = y.long()
        emb = self(x)   # przepuszczenie danych przez model: self.model(x) -> ReIDBackbone(x)
                        # wyjście: embeddingi

        # miner po warmupie
        use_miner = (
            self.current_epoch >= self.miner_warmup
            and self.miner is not None
        )

        if use_miner:
            hard = self.miner(emb, y)   # przeszukuje batch i wybiera hard positives (te same osoby, które są daleko) i hard negatives (różne osoby, które są blisko)
            loss = self.loss_fn(emb, y, hard)   # liczy stratę tylko dla trudnych przypadków
        else:
            loss = self.loss_fn(emb, y) # loss samodzielnie dobiera pary negatywne i pozytywne w batchu (wszystkie kombinacje)

        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss # przy odbieraniu loss wywyływane jest loss.backward i optimizer.step

    def validation_step(self, batch, batch_idx):
        # walidacja bez minera
        '''
            | Zmienna | Znaczenie              | Typ               |
            | ------- | ---------------------- | ----------------- |
            | x     | batch obrazów          | tensor [B, 3, H, W] |
            | y     | etykiety ID zawodników | tensor [B]          |
            | act   | indeksy akcji          | tensor [B]          |
            | path  | ścieżki do plików      | lista stringów      |

        '''
        x, y, act, path = batch
        y = y.long()
        emb = self(x)   # wywołanie forward()
        loss = self.loss_fn(emb, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    # Wybór optymalizatora i schedulera uczenia
    def configure_optimizers(self):
        # optymalizator AdamW
        opt = torch.optim.AdamW(self.parameters(), lr=CFG.lr)

        # Cosine Annealing LR
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=CFG.epochs
        )
        # Linear Warmup
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=2
        )
        # połączenie obu schedulerów
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cos], milestones=[2] # po 2 epokach kończy się warmup i scheduler przechodzi w cosine
        )
        return {"optimizer": opt, "lr_scheduler": sched}


# ===================== DATALOADERS =====================
def make_loaders():
    # dataset train i valid
    train_ds = SoccerNetReID(CFG.data_root, split="train")
    valid_ds = SoccerNetReID(CFG.data_root, split="valid")

    '''
    PK SAMPLER per-akcja
        - P liczba klas ID w batchu
        - K liczba przykładów na klasę
        - batch_size = P x K
        - każdy batch pochodzi z jednej akcji
    '''
    K = CFG.K   # liczba próbek na klasę
    bs = CFG.batch_size  # rozmiar batcha

    sampler = ActionPKSampler(
        labels=train_ds.labels,
        actions=train_ds.actions,
        action_to_indices=train_ds.action_to_indices,
        batch_size=bs,
        K=K,
    )

    '''
    Wspólne parametry dla wszystkich loaderów

    | parametr             | opis                          |
    | -------------------- | ----------------------------- |
    | num_workers          | liczba wątków do ładowania    |
    | pin_memory           | szybki transfer CPU→GPU (CUDA)|
    | drop_last=True       | odrzuca niepełny ostatni batch|
    | persistent_workers   | utrzymuje wątki między epokami
    '''
    common = dict(
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(CFG.num_workers > 0),
    )

    # loader train z naszym samplerem akcyjnym
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, **common
    )

    # loader valid bez shuffle i bez PK samplera
    val_loader = DataLoader(
        valid_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    print(f"train items: {len(train_ds)} | valid items: {len(valid_ds)}")
    return train_loader, val_loader


# ===================== TRAIN =====================
def main():
    import platform, multiprocessing as mp

    # Windows wymaga spawn
    if platform.system() == "Windows":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # wybór urządzenia
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"      # GPU Apple
    else:
        accelerator = "cpu"

    # seedy
    set_seed(CFG.seed)
    torch.set_float32_matmul_precision("medium")

    # cudnn tylko dla kart CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # loadery
    train_loader, val_loader = make_loaders()

    # model
    model = ReIDLit(CFG)

    # checkpoint co epokę
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch_{epoch}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    # logger CSV
    logger = CSVLogger("runs", name="reid_full")

    # precyzja dla MPS
    precision = CFG.precision
    if accelerator == "mps":
        precision = "32-true"   # bez AMP na MPS

    '''
    Trener Lightning
    Ważne parametry:
        - accelerator - gpu, mps, cpu
        - devices=1 - jedno urządzenie
        - max_epochs - liczba epok
        - precision - 16-mixed lub 32
        - callbacks - checkpointy
        - limit_val_batches=1.0 - pełna walidacja
        - num_sanity_val_steps=2 - test walidacji przed startem
        - logger - CSV logger
    Lightning sam obsługuje:
        - backward
        - optimizer
        - scheduler
        - logi
        - checkpointy
    '''
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=CFG.epochs,
        precision=precision,
        log_every_n_steps=20,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        limit_val_batches=1.0,
        num_sanity_val_steps=2,
        logger=logger,
    )
    # start treningu
    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    main()
