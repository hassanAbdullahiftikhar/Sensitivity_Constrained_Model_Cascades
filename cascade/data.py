"""Dataset loading: PathMNIST (MedMNIST+, larger resolutions) and HAM10000 with lesion-level splits.

Images are returned as uint8 arrays (N, H, W, 3) and int64 labels (N,). Everything is held in RAM;
at 128 px PathMNIST needs ~5.3 GB, so no DataLoader workers are used (Windows spawn would copy it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("CASCADE_DATA", Path(__file__).resolve().parent.parent / "data"))


@dataclass
class Split:
    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    class_names: tuple[str, ...]
    positive: tuple[int, ...]  # class indices counted as "positive" for sensitivity/specificity

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# NCT-CRC-HE tissue classes in MedMNIST label order; positive = colorectal adenocarcinoma epithelium.
PATHMNIST = DatasetSpec(
    "pathmnist",
    ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"),
    (8,),
)
# HAM10000 diagnoses (alphabetical); positive = malignant / pre-malignant: akiec, bcc, mel.
HAM10000 = DatasetSpec(
    "ham10000",
    ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
    (0, 1, 4),
)
# Same data and split as HAM10000; trained with class-balanced loss (see train.py) as a robustness check.
HAM10000_BAL = DatasetSpec("ham10000bal", HAM10000.class_names, HAM10000.positive)
SPECS = {s.name: s for s in (PATHMNIST, HAM10000, HAM10000_BAL)}


def load(name: str, size: int = 128) -> dict[str, Split]:
    if name == "pathmnist":
        return load_pathmnist(size)
    if name in ("ham10000", "ham10000bal"):
        return load_ham10000(size)
    raise ValueError(f"unknown dataset {name!r}")


def load_pathmnist(size: int = 128) -> dict[str, Split]:
    root = DATA_ROOT / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("pathmnist.npz" if size == 28 else f"pathmnist_{size}.npz")
    if not path.exists():
        import medmnist

        medmnist.PathMNIST(split="test", download=True, size=size, root=str(root))
    z = np.load(path)
    return {
        s: Split(z[f"{s}_images"], z[f"{s}_labels"].reshape(-1).astype(np.int64))
        for s in ("train", "val", "test")
    }


def load_ham10000(size: int = 128, split_seed: int = 0) -> dict[str, Split]:
    """HAM10000 resized to size x size, split ~72/14/14 by lesion_id (no lesion appears in two splits).

    Expects the Harvard Dataverse files (doi:10.7910/DVN/DBW86T) unpacked under data/ham10000/:
    HAM10000_metadata(.csv|.tab) and the HAM10000_images_part_{1,2} jpgs anywhere below that folder.
    """
    root = DATA_ROOT / "ham10000"
    cache = root / f"ham10000_{size}_split{split_seed}.npz"
    if not cache.exists():
        _build_ham_cache(root, cache, size, split_seed)
    z = np.load(cache)
    return {s: Split(z[f"{s}_images"], z[f"{s}_labels"]) for s in ("train", "val", "test")}


def _build_ham_cache(root: Path, cache: Path, size: int, split_seed: int) -> None:
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import StratifiedGroupKFold

    meta_path = next((p for p in root.rglob("HAM10000_metadata*") if p.suffix in {".csv", ".tab", ""}), None)
    if meta_path is None:
        raise FileNotFoundError(f"HAM10000 metadata not found under {root}")
    meta = pd.read_csv(meta_path, sep="\t" if meta_path.suffix == ".tab" else ",")
    paths = {p.stem: p for p in root.rglob("ISIC_*.jpg")}
    meta = meta[meta["image_id"].isin(paths)].reset_index(drop=True)
    labels = meta["dx"].map({c: i for i, c in enumerate(HAM10000.class_names)}).to_numpy(np.int64)
    groups = meta["lesion_id"].to_numpy()

    images = np.empty((len(meta), size, size, 3), np.uint8)
    for i, image_id in enumerate(meta["image_id"]):
        im = Image.open(paths[image_id]).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        images[i] = np.asarray(im.resize((size, size), Image.BICUBIC))

    idx = np.arange(len(meta))
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=split_seed)
    rest, test = next(outer.split(idx, labels, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=split_seed)
    tr, va = next(inner.split(rest, labels[rest], groups[rest]))
    train, val = rest[tr], rest[va]
    assert not (set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]))

    np.savez(
        cache,
        **{f"{n}_images": images[ix] for n, ix in (("train", train), ("val", val), ("test", test))},
        **{f"{n}_labels": labels[ix] for n, ix in (("train", train), ("val", val), ("test", test))},
    )
