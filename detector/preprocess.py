from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    resize_enabled: bool = True
    resize_width: int = 960
    gaussian_blur_enabled: bool = False
    gaussian_kernel: int = 3
    contrast_enabled: bool = True
    contrast_alpha: float = 1.08
    contrast_beta: int = 2
    denoise_enabled: bool = False
    denoise_strength: int = 5


class ImagePreprocessor:
    """Pipeline simples e reutilizável de pré-processamento antes da inferência YOLO."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    @staticmethod
    def _odd_kernel(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame vazio recebido pelo pré-processador.")

        processed = frame_bgr.copy()
        cfg = self.config

        if cfg.resize_enabled and cfg.resize_width > 0 and processed.shape[1] != cfg.resize_width:
            scale = cfg.resize_width / processed.shape[1]
            height = max(1, int(processed.shape[0] * scale))
            processed = cv2.resize(processed, (cfg.resize_width, height), interpolation=cv2.INTER_AREA)

        if cfg.gaussian_blur_enabled:
            k = self._odd_kernel(cfg.gaussian_kernel)
            processed = cv2.GaussianBlur(processed, (k, k), 0)

        if cfg.denoise_enabled:
            strength = max(1, int(cfg.denoise_strength))
            processed = cv2.fastNlMeansDenoisingColored(
                processed, None, strength, strength, 7, 21
            )

        if cfg.contrast_enabled:
            processed = cv2.convertScaleAbs(
                processed,
                alpha=max(0.1, float(cfg.contrast_alpha)),
                beta=int(cfg.contrast_beta),
            )

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        return processed, {"rgb": rgb, "hsv": hsv}
