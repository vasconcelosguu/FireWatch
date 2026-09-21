from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "fire_model.pt"
DATA_YAML = ROOT / "datasets" / "fire_smoke" / "data.yaml"
RUNTIME_YAML = ROOT / "reports" / "data_runtime.yaml"


def make_runtime_yaml() -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida o modelo FireWatch AI.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError("Modelo personalizado ainda não treinado. Execute scripts/train_model.py.")

    from ultralytics import YOLO

    runtime_yaml = make_runtime_yaml()
    model = YOLO(str(args.model))
    metrics = model.val(data=str(runtime_yaml), split="test", imgsz=args.imgsz, conf=args.confidence, plots=True)
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("A validação não retornou métricas de bounding boxes.")

    print("\n=== MÉTRICAS FIREWATCH AI ===")
    print(f"mAP50:     {float(getattr(box, 'map50', 0.0)):.4f}")
    print(f"mAP50-95:  {float(getattr(box, 'map', 0.0)):.4f}")
    print(f"Precision: {float(getattr(box, 'mp', 0.0)):.4f}")
    print(f"Recall:    {float(getattr(box, 'mr', 0.0)):.4f}")


if __name__ == "__main__":
    main()
