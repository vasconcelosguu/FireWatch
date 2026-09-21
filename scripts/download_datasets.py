from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_sources"
DFIRE_DIR = RAW_DIR / "dfire"
INDOOR_DIR = RAW_DIR / "indoor_fire_smoke"

# Public D-Fire mirror on Hugging Face (fallback) and the public Kaggle copy
# linked from the official D-Fire repository.
HF_REPO = "hanvithSai/gavin-dfire-raw"
KAGGLE_DATASET = "sayedgamal99/smoke-fire-detection-yolo"
INDOOR_URL = "https://zenodo.org/records/15826133/files/Indoor%20Fire%20Smoke.zip?download=1"


def _find_dataset_root(root: Path) -> Path | None:
    """Find a directory containing the expected D-Fire train/images layout."""
    candidates = [root, *[p for p in root.rglob("*") if p.is_dir()]]
    for candidate in candidates:
        if (candidate / "train" / "images").is_dir() and (candidate / "train" / "labels").is_dir():
            return candidate
        # Some copies use train/images but keep labels in a sibling directory.
        if (candidate / "train" / "images").is_dir():
            return candidate
    return None


def _write_source(path: Path) -> None:
    DFIRE_DIR.mkdir(parents=True, exist_ok=True)
    (DFIRE_DIR / ".source_path").write_text(str(path.resolve()), encoding="utf-8")


def _has_dfire() -> bool:
    source_file = DFIRE_DIR / ".source_path"
    if source_file.exists():
        source = Path(source_file.read_text(encoding="utf-8").strip())
        if source.exists() and _find_dataset_root(source):
            return True
    return bool(_find_dataset_root(DFIRE_DIR))


def download_dfire_kaggle() -> Path:
    """Prefer the public Kaggle copy: avoids the large HF Xet timeout path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub não está instalado.") from exc

    print(f"[D-Fire] Tentando Kaggle: {KAGGLE_DATASET}")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=False))
    dataset_root = _find_dataset_root(path)
    if dataset_root is None:
        raise RuntimeError(f"Download do Kaggle terminou, mas não encontrei train/images em: {path}")
    _write_source(dataset_root)
    print(f"[OK] D-Fire encontrado em: {dataset_root}")
    return dataset_root


def download_dfire_huggingface() -> Path:
    """Fallback with long timeouts, Xet support and resumable downloads."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub não está instalado.") from exc

    DFIRE_DIR.mkdir(parents=True, exist_ok=True)

    # The default HF HTTP timeout is too aggressive for multi-GB archives.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"[D-Fire] Fallback Hugging Face: {HF_REPO}")
    print("        Timeout de download: 180s; downloads são retomáveis.")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DFIRE_DIR),
        allow_patterns=["train.tar", "test.tar"],
        resume_download=True,
    )

    # Do not re-download/re-extract archives that already exist.
    for archive_name in ("train.tar", "test.tar"):
        archive = DFIRE_DIR / archive_name
        if not archive.exists():
            continue
        target = DFIRE_DIR / archive.stem
        if _find_dataset_root(target):
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"  extraindo {archive_name}...")
        import tarfile
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")

    dataset_root = _find_dataset_root(DFIRE_DIR)
    if dataset_root is None:
        raise RuntimeError("D-Fire foi baixado, mas a estrutura train/images não foi encontrada.")
    _write_source(dataset_root)
    return dataset_root


def download_dfire() -> Path:
    if _has_dfire():
        source = Path((DFIRE_DIR / ".source_path").read_text(encoding="utf-8").strip()) if (DFIRE_DIR / ".source_path").exists() else DFIRE_DIR
        print(f"[OK] D-Fire já disponível em: {source}")
        return source

    DFIRE_DIR.mkdir(parents=True, exist_ok=True)

    # Kaggle is tried first because the HF mirror used previously was timing
    # out on the 2.6 GB train.tar and 640 MB test.tar files.
    try:
        return download_dfire_kaggle()
    except Exception as kaggle_error:
        print(f"[AVISO] Kaggle não funcionou: {kaggle_error}")
        print("[INFO] Tentando D-Fire pelo Hugging Face com timeout maior...")
        return download_dfire_huggingface()


def download_indoor() -> Path:
    import requests

    INDOOR_DIR.mkdir(parents=True, exist_ok=True)
    archive = INDOOR_DIR / "Indoor Fire Smoke.zip"
    marker = INDOOR_DIR / ".downloaded"
    if marker.exists():
        print("[OK] Indoor Fire Smoke já foi baixado.")
        return INDOOR_DIR

    print("Baixando Indoor Fire Smoke Dataset...")
    temp = archive.with_suffix(".part")
    headers = {"User-Agent": "FireWatch-AI/2.0"}
    with requests.get(INDOOR_URL, stream=True, timeout=(30, 180), headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        received = 0
        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r  {received / total * 100:5.1f}%", end="", flush=True)
    temp.replace(archive)
    print()

    extract_dir = INDOOR_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)
    marker.write_text("Indoor Fire Smoke downloaded and extracted.\n", encoding="utf-8")
    return INDOOR_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa automaticamente datasets públicos de fogo/fumaça.")
    parser.add_argument("--datasets", nargs="+", choices=["dfire", "indoor", "all"], default=["dfire"])
    args = parser.parse_args()
    selected = {"dfire", "indoor"} if "all" in args.datasets else set(args.datasets)

    if "dfire" in selected:
        download_dfire()
    if "indoor" in selected:
        download_indoor()

    print("\nDownloads concluídos. Próximo passo: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
