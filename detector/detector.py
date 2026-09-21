from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FireSmokeDetector:
    """Encapsula carregamento, inferência e desenho do modelo Fire/Smoke."""

    CLASS_NAMES = {0: "Fire", 1: "Smoke"}
    COLORS = {
        0: (35, 35, 235),   # vermelho (BGR)
        1: (165, 125, 85),  # azul/cinza (BGR)
    }

    def __init__(self, model_path: str | Path = "models/fire_model.pt", confidence: float = 0.50) -> None:
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.model: Any | None = None
        self.error_message: str | None = None
        self.load_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def load_model(self) -> bool:
        self.model = None
        self.error_message = None

        if not self.model_path.exists():
            self.error_message = "Modelo personalizado ainda não treinado."
            return False

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depende do ambiente
            self.error_message = f"Ultralytics não está disponível: {exc}"
            return False

        try:
            self.model = YOLO(str(self.model_path))
            return True
        except Exception as exc:
            self.error_message = f"Falha ao carregar o modelo: {exc}"
            self.model = None
            return False

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if not self.available:
            return []

        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            verbose=False,
        )
        return self.get_detections(results)

    def get_detections(self, results: Any) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                except Exception:
                    continue

                if class_id not in self.CLASS_NAMES:
                    continue

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.CLASS_NAMES[class_id],
                        confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections

    def draw_detections(self, frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        for det in detections:
            x1 = max(0, min(w - 1, det.x1))
            y1 = max(0, min(h - 1, det.y1))
            x2 = max(0, min(w - 1, det.x2))
            y2 = max(0, min(h - 1, det.y2))
            color = self.COLORS.get(det.class_id, (255, 255, 255))

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence * 100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y1 = max(0, y1 - text_h - baseline - 8)
            cv2.rectangle(
                output,
                (x1, label_y1),
                (min(w - 1, x1 + text_w + 12), y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 6, max(text_h + 2, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return output
