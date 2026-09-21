from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "fire_model.pt"
DATA = ROOT / "reports" / "data_runtime.yaml"
OUT = ROOT / "reports" / "evaluation_summary.csv"


def main():
    if not MODEL.exists():
        raise FileNotFoundError("models/fire_model.pt não existe. Treine o modelo primeiro.")
    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    rows = []
    for conf in (0.25, 0.40, 0.50, 0.60, 0.75):
        metrics = model.val(data=str(DATA), split="test", imgsz=640, conf=conf, plots=False, verbose=False)
        box = metrics.box
        rows.append({
            "confidence": conf,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Experimento de threshold salvo em {OUT}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
