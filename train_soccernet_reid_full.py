import os
import re
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
Klasa przygotowująca dane dla sieci.
    - Znaleźć wszystkie obrazy w folderze (np. train)
    - Sparsować ich nazwy, by znać kto to i z jakiej akcji
    - Nadać unikalne etykiety liczbowo (labele)
    - Przygotować dane + augmentacje
'''
class SoccerNetReID(Dataset):
    # dataset do ReID
    # label = action_idx + person_uid
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split  # Podfolder
        split_dir = self.root / split   # Pełna ścieżka (np. data/reid-2023/train)
        if not split_dir.exists():
            raise FileNotFoundError(f"brak katalogu: {split_dir}")

        all_imgs = list_images(split_dir)   # Przeszukiwanie plików po podfolderach (np. england_epl/2015-2016/...)

        # parsowane nazw plików
        samples = []
        for p in all_imgs:
            meta = parse_filename(p.stem)
            if meta is None:
                continue
            key = f"{meta['action_idx']}|{meta['person_uid']}"  # Unikalny id osoby w konkretnej akcji
            samples.append((str(p), key, int(meta["action_idx"])))  # Tworzenie listy trójek ścieżka, klucz, akcja (np. ("path/to/img.png", "12|3456", 12)).

        # remap klucza na int "12|3456" -> 0, "12|7890" -> 1. Wynik (path, label_int, action_idx)
        label_to_int = {}
        items = []
        for path, key, act in samples:
            if key not in label_to_int:
                label_to_int[key] = len(label_to_int)
            items.append((path, label_to_int[key], act))

        # zapis pól do klasy
        self.items = items  # Główna lista z danymi
        self.labels = [y for _, y, _ in items]  # Lista etykiet liczbowych (używana przez sampler PK)
        self.actions = [a for *_, a in items]   # Lista indeksów akcji

        # transformacje
        if transform is None:
            transform = build_transform(train=(split == "train"))
        self.transform = transform

    '''
    Zwraca liczbę wszystkich próbek, by DataLoader widział ile kroków ma epoka
    '''
    def __len__(self):
        return len(self.items)

    '''
    - wybiera rekord po indeksie
    - otwiera obraz z dysku przez PIL.Image.open
    - konwertuje do RGB (na wypadek, gdyby był grayscale)
    - stosuje transformacje (augmentacje lub normalizację)
    
    Zwraca:
    - img: tensor 3xHxW
    - y: etykieta (int, ID gracza)
    - act: numer akcji (do analizy kontekstu)
    - path: ścieżka (przydatna np. do ewaluacji i logów)
    '''
    def __getitem__(self, idx):
        path, y, act = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y, act, path


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
    PK SAMPLER, jeśli możliwy
        - P liczba klas - unikatowych zawodników w batchu
        - K liczba przykładów (obrazów) na klasę
        - batch_size = P x K
    '''
    K = CFG.K   # liczba próbek na klasę
    bs = CFG.batch_size # rozmiar batcha
    sampler = None
    use_pk = False
    try:
        assert bs % K == 0  # sprawdzenie, czy batch dzieli się na K
        '''
            - train_ds.labels - lista etykiet z datasetu ([0, 0, 1, 1, 1, 2, 3, ...])
            - m - liczba prrzykładów na klasę ile ma wziąć z datasetu
            - length_before_new_iter - długość datasetu (ile iteracji zanim sampler się odświezy)
        '''
        sampler = MPerClassSampler(
            train_ds.labels, m=K, batch_size=bs,
            length_before_new_iter=len(train_ds)
        )
        use_pk = True
    except Exception as e:
        print("PK sampler wyłączony:", e)   # jeśli błąd to zamiast samplera uzywamy shullfe=True

    '''
    Wspólne parametry dla wszystkich loaderów
    | parametr             | opis                                                         |
    | -------------------- | ------------------------------------------------------------ |
    | `num_workers`        | liczba wątków do ładowania danych (4 → 4 równoległe wątki)   |
    | `pin_memory=True`    | przyspiesza transfer danych CPU→GPU                          |
    | `drop_last=True`     | odrzuca ostatni batch, jeśli niepełny                        |
    | `persistent_workers` | utrzymuje procesy robocze między epokami (oszczędność czasu) |

    '''
    common = dict(
        num_workers=CFG.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(CFG.num_workers > 0),
    )

    # loader train z samplerem lub shuffle
    if use_pk:
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler, **common
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, **common
        )

    # loader valid bez shuffle i bez PK Samplera
    val_loader = DataLoader(
        valid_ds, batch_size=bs, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
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
