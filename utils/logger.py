from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMNS = [
    "timestamp",
    "source",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "image_path",
]


class DetectionLogger:
    def __init__(self, csv_path: str | Path = "logs/detections.csv") -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.ensure_file()

    def ensure_file(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()

    def log(
        self,
        source: str,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_path: str,
        timestamp: datetime | None = None,
    ) -> None:
        row = {
            "timestamp": (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "class_name": class_name,
            "confidence": f"{float(confidence):.6f}",
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "image_path": image_path,
        }
        with self._lock:
            self.ensure_file()
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=COLUMNS)
                writer.writerow(row)

    def load(self) -> pd.DataFrame:
        self.ensure_file()
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=COLUMNS)

        for column in COLUMNS:
            if column not in df.columns:
                df[column] = None
        if not df.empty:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df[COLUMNS]


def basic_statistics(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "total": 0,
            "fire": 0,
            "smoke": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
        }
    return {
        "total": int(len(df)),
        "fire": int((df["class_name"] == "Fire").sum()),
        "smoke": int((df["class_name"] == "Smoke").sum()),
        "mean_confidence": float(df["confidence"].mean()),
        "max_confidence": float(df["confidence"].max()),
    }
