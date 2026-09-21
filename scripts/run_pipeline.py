from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo: download -> preparação -> treino -> avaliação.")
    parser.add_argument("--sources", nargs="+", choices=["dfire", "indoor"], default=["indoor"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="0 para NVIDIA GPU; cpu para CPU")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--max-images", type=int, default=1500, help="Limite total de imagens preparadas; 0 usa tudo.")
    args = parser.parse_args()

    run("download_datasets.py", "--datasets", *args.sources)
    prepare_args = ["--sources", *args.sources, "--max-images", str(args.max_images)]
    run("prepare_dataset.py", *prepare_args)
    train_args = ["--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--batch", str(args.batch), "--model", args.model]
    if args.device:
        train_args += ["--device", args.device]
    run("train_model.py", *train_args)
    run("validate_model.py")
    run("evaluate_experiments.py")

    print("\nPIPELINE CONCLUÍDO. Modelo: models/fire_model.pt")


if __name__ == "__main__":
    main()
