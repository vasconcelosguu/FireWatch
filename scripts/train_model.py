from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "fire_smoke" / "data.yaml"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs" / "firewatch"
RUNTIME_YAML = ROOT / "reports" / "data_runtime.yaml"


def validate_dataset_structure() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {DATA_YAML}")
    for split in ["train", "val", "test"]:
        images = ROOT / "datasets" / "fire_smoke" / split / "images"
        labels = ROOT / "datasets" / "fire_smoke" / split / "labels"
        if not images.exists() or not labels.exists():
            raise FileNotFoundError(f"Estrutura ausente no split {split}.")


def make_runtime_yaml() -> Path:
    """Mantém o data.yaml acadêmico pedido e gera uma cópia com path absoluto para a Ultralytics."""
    dataset_root = (ROOT / "datasets" / "fire_smoke").resolve().as_posix()
    RUNTIME_YAML.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_YAML.write_text(
        f"path: {dataset_root}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: Fire\n"
        "  1: Smoke\n",
        encoding="utf-8",
    )
    return RUNTIME_YAML


def load_training_model(preferred: str | None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics não instalado. Execute: python -m pip install -r requirements.txt") from exc

    candidates = [preferred] if preferred else ["yolo11n.pt", "yolov8n.pt"]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            print(f"Tentando pesos base: {candidate}")
            return YOLO(candidate), candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(
        "Nenhum peso pré-treinado compatível pôde ser carregado. "
        "Atualize a Ultralytics ou informe --model com um peso suportado.\n" + detail
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina detector Fire/Smoke com Ultralytics YOLO.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.50, help="Threshold usado na validação pós-treino.")
    parser.add_argument("--model", type=str, default=None, help="Peso base, ex.: yolo26n.pt")
    parser.add_argument("--device", type=str, default=None, help="Ex.: 0 para GPU ou cpu")
    args = parser.parse_args()

    validate_dataset_structure()
    runtime_yaml = make_runtime_yaml()
    model, base_name = load_training_model(args.model)
    print(f"Modelo base selecionado: {base_name}")

    train_kwargs = {
        "data": str(runtime_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(RUNS_DIR),
        "name": "train",
        "exist_ok": True,
        "plots": True,
        "patience": 5,
        "workers": 2,
        "cache": False,
        "degrees": 5.0,
        "translate": 0.10,
        "scale": 0.50,
        "fliplr": 0.50,
        "mosaic": 1.0,
        "mixup": 0.10,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    best = RUNS_DIR / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Treinamento terminou, mas best.pt não foi encontrado em {best}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "fire_model.pt"
    shutil.copy2(best, target)
    print(f"Melhor modelo copiado para: {target}")

    trained = type(model)(str(target))
    metrics = trained.val(data=str(runtime_yaml), imgsz=args.imgsz, conf=args.confidence)
    box = getattr(metrics, "box", None)
    if box is not None:
        print(f"mAP50: {getattr(box, 'map50', float('nan')):.4f}")
        print(f"mAP50-95: {getattr(box, 'map', float('nan')):.4f}")
        print(f"Precision: {getattr(box, 'mp', float('nan')):.4f}")
        print(f"Recall: {getattr(box, 'mr', float('nan')):.4f}")


if __name__ == "__main__":
    main()
