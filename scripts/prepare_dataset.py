from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "fire_smoke"
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo_label(path: Path, source: str):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id, x, y, w, h = int(parts[0]), *map(float, parts[1:])
        except ValueError:
            continue
        # D-Fire ships 0=smoke, 1=fire. FireWatch uses 0=fire, 1=smoke.
        if source == "dfire":
            if class_id == 0:
                class_id = 1
            elif class_id == 1:
                class_id = 0
            else:
                continue
        elif source == "indoor":
            # Common indoor dataset convention is already fire=0/smoke=1.
            if class_id not in (0, 1):
                continue
        rows.append((class_id, x, y, w, h))
    return rows


def find_matching_label(image: Path):
    candidates = [image.with_suffix(".txt"), image.with_suffix(".xml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Some extracted datasets keep labels in a sibling labels directory.
    for parent in [image.parent, *image.parents]:
        labels_dir = parent / "labels"
        candidate = labels_dir / f"{image.stem}.txt"
        if candidate.exists():
            return candidate
        candidate = labels_dir / f"{image.stem}.xml"
        if candidate.exists():
            return candidate
    return None


def parse_voc(path: Path):
    rows = []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return rows
    size = root.find("size")
    if size is None:
        return rows
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return rows
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if "fire" in name and "smoke" not in name:
            class_id = 0
        elif "smoke" in name:
            class_id = 1
        else:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        x1 = float(box.findtext("xmin", "0"))
        y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0"))
        y2 = float(box.findtext("ymax", "0"))
        x = ((x1 + x2) / 2) / width
        y = ((y1 + y2) / 2) / height
        w = abs(x2 - x1) / width
        h = abs(y2 - y1) / height
        rows.append((class_id, x, y, w, h))
    return rows


def parse_annotations(image: Path, source: str):
    label = find_matching_label(image)
    if label is None:
        return []
    if label.suffix.lower() == ".txt":
        return read_yolo_label(label, source)
    return parse_voc(label)


def collect_source(source: str, raw_root: Path):
    records = []
    for image in image_files(raw_root):
        rows = parse_annotations(image, source)
        records.append({"image": image, "rows": rows, "source": source})
    return records


def safe_copy(record, split: str, index: int, used_hashes: set[str]):
    image = record["image"]
    digest = sha256(image)
    if digest in used_hashes:
        return None
    used_hashes.add(digest)

    out_img_dir = OUTPUT / split / "images"
    out_lbl_dir = OUTPUT / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{record['source']}_{index:06d}"
    destination = out_img_dir / f"{stem}{image.suffix.lower()}"
    shutil.copy2(image, destination)
    label_path = out_lbl_dir / f"{stem}.txt"
    lines = [
        f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for cid, x, y, w, h in record["rows"]
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def split_dfire(records):
    # D-Fire official test stays untouched. The original train pool is split into train/val.
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    val_count = max(1, int(len(records) * 0.15))
    return records[val_count:], records[:val_count]


def build(sources: list[str], max_images: int | None = None):
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    used_hashes: set[str] = set()
    manifest = {"sources": sources, "seed": RANDOM_SEED, "splits": {}}
    all_train, all_val, all_test = [], [], []

    if "dfire" in sources:
        raw = ROOT / "data_sources" / "dfire"
        source_file = raw / ".source_path"
        if source_file.exists():
            candidate = Path(source_file.read_text(encoding="utf-8").strip())
            if candidate.exists():
                raw = candidate
        train_root = raw / "train"
        test_root = raw / "test"
        if not train_root.exists():
            raise FileNotFoundError("D-Fire não encontrado. Rode scripts/download_datasets.py --datasets dfire")
        df_train = collect_source("dfire", train_root)
        df_test = collect_source("dfire", test_root) if test_root.exists() else []
        tr, va = split_dfire(df_train)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(df_test)

    if "indoor" in sources:
        raw = ROOT / "data_sources" / "indoor_fire_smoke" / "extracted"
        if not raw.exists():
            raise FileNotFoundError("Indoor dataset não encontrado. Rode scripts/download_datasets.py --datasets indoor")
        indoor = collect_source("indoor", raw)
        random.seed(RANDOM_SEED)
        random.shuffle(indoor)
        n = len(indoor)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        all_train.extend(indoor[:n_train])
        all_val.extend(indoor[n_train:n_train + n_val])
        all_test.extend(indoor[n_train + n_val:])

    random.seed(RANDOM_SEED)
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)

    # Modo acadêmico leve: limita o total de imagens, preservando a proporção 70/15/15.
    # O dataset completo continua disponível; apenas uma amostra reprodutível é usada no treino.
    if max_images is not None and max_images > 0:
        total = len(all_train) + len(all_val) + len(all_test)
        if total > max_images:
            n_train = round(max_images * 0.70)
            n_val = round(max_images * 0.15)
            n_test = max_images - n_train - n_val
            all_train = all_train[:min(n_train, len(all_train))]
            all_val = all_val[:min(n_val, len(all_val))]
            all_test = all_test[:min(n_test, len(all_test))]
            print(f"Modo leve: usando {len(all_train)+len(all_val)+len(all_test)} imagens de {total} disponíveis.")

    counters = {"train": 0, "val": 0, "test": 0}
    for split, records in [("train", all_train), ("val", all_val), ("test", all_test)]:
        for record in records:
            result = safe_copy(record, split, counters[split], used_hashes)
            if result:
                counters[split] += 1
        manifest["splits"][split] = counters[split]

    data_yaml = OUTPUT / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: Fire\n"
        "  1: Smoke\n",
        encoding="utf-8",
    )
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== DATASET FIREWATCH ===")
    print(json.dumps(manifest, indent=2))
    print(f"Dataset pronto em: {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Normaliza datasets públicos para o formato YOLO do FireWatch AI.")
    parser.add_argument("--sources", nargs="+", choices=["dfire", "indoor"], default=["dfire"])
    parser.add_argument("--max-images", type=int, default=1500, help="Limite total de imagens preparadas; 0 usa tudo.")
    args = parser.parse_args()
    build(args.sources, None if args.max_images <= 0 else args.max_images)


if __name__ == "__main__":
    main()
