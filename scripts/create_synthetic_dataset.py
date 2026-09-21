from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "fire_smoke"
WIDTH = 640
HEIGHT = 384


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def yolo(self, width: int, height: int) -> str:
        x_center = ((self.x1 + self.x2) / 2) / width
        y_center = ((self.y1 + self.y2) / 2) / height
        box_width = (self.x2 - self.x1) / width
        box_height = (self.y2 - self.y1) / height
        values = [x_center, y_center, box_width, box_height]
        values = [min(1.0, max(0.0, value)) for value in values]
        return f"{self.class_id} " + " ".join(f"{v:.6f}" for v in values)


def _gradient_background(rng: random.Random, night: bool) -> np.ndarray:
    top = np.array((20, 28, 48) if night else (185, 205, 220), dtype=np.float32)
    bottom = np.array((38, 42, 50) if night else (95, 120, 105), dtype=np.float32)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        image[y, :, :] = top * (1 - t) + bottom * t
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, 5.0, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image


def _draw_scene(image: np.ndarray, rng: random.Random, scene: str, night: bool) -> None:
    if scene == "forest":
        for _ in range(rng.randint(12, 28)):
            x = rng.randint(0, WIDTH - 1)
            trunk_h = rng.randint(50, 150)
            base = HEIGHT - rng.randint(0, 20)
            cv2.rectangle(image, (x, base - trunk_h), (x + rng.randint(3, 8), base), (45, 55, 60), -1)
            radius = rng.randint(16, 38)
            foliage = (30, rng.randint(70, 115), rng.randint(35, 65))
            cv2.circle(image, (x + 2, max(20, base - trunk_h)), radius, foliage, -1)
    elif scene == "residential":
        x = rng.randint(80, 360)
        y = rng.randint(145, 215)
        w = rng.randint(160, 230)
        h = rng.randint(90, 135)
        wall = (110, 135, 155) if night else (170, 185, 195)
        cv2.rectangle(image, (x, y), (min(WIDTH - 1, x + w), min(HEIGHT - 1, y + h)), wall, -1)
        roof = np.array([[x - 20, y], [x + w // 2, y - 70], [x + w + 20, y]], dtype=np.int32)
        cv2.fillPoly(image, [roof], (55, 55, 70))
        for wx in range(x + 30, x + w - 20, 65):
            cv2.rectangle(image, (wx, y + 30), (wx + 28, y + 62), (70, 88, 100), -1)
    elif scene == "indoor":
        image[:] = cv2.addWeighted(image, 0.35, np.full_like(image, (115, 110, 105)), 0.65, 0)
        cv2.rectangle(image, (0, 285), (WIDTH, HEIGHT), (65, 70, 75), -1)
        cv2.line(image, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), (85, 85, 90), 3)
        cv2.rectangle(image, (70, 205), (260, 295), (78, 82, 92), -1)
        cv2.rectangle(image, (430, 175), (560, 290), (72, 78, 85), -1)
    elif scene == "camp":
        cv2.rectangle(image, (0, 295), (WIDTH, HEIGHT), (70, 92, 78), -1)
        for _ in range(10):
            x = rng.randint(30, WIDTH - 30)
            cv2.line(image, (x, 310), (x + rng.randint(-15, 15), 350), (55, 45, 40), 5)
    else:
        cv2.rectangle(image, (0, 290), (WIDTH, HEIGHT), (90, 95, 100), -1)

    if night:
        overlay = np.zeros_like(image)
        image[:] = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)


def _draw_fire(image: np.ndarray, rng: random.Random, location: str = "random") -> Box:
    bw = rng.randint(85, 190)
    bh = rng.randint(95, 220)
    x1 = rng.randint(25, WIDTH - bw - 25)
    y2 = rng.randint(int(HEIGHT * 0.70), HEIGHT - 15)
    y1 = max(15, y2 - bh)
    x2 = x1 + bw

    glow = np.zeros_like(image)
    center = ((x1 + x2) // 2, y2 - bh // 3)
    for radius, color, alpha in [
        (int(bw * 0.65), (0, 40, 170), 0.12),
        (int(bw * 0.45), (0, 80, 235), 0.16),
    ]:
        cv2.circle(glow, center, max(8, radius), color, -1)
        image[:] = cv2.addWeighted(image, 1.0, glow, alpha, 0)
        glow[:] = 0

    flame_layer = image.copy()
    for _ in range(rng.randint(6, 12)):
        base_x = rng.randint(x1 + 8, x2 - 8)
        base_y = rng.randint(y2 - 22, y2 - 4)
        half = rng.randint(10, max(12, bw // 5))
        tip_y = rng.randint(y1 + 4, max(y1 + 6, y2 - 45))
        points = np.array(
            [
                [max(x1 + 1, base_x - half), base_y],
                [base_x, tip_y],
                [min(x2 - 1, base_x + half), base_y],
            ],
            dtype=np.int32,
        )
        color = rng.choice([(0, 70, 255), (0, 120, 255), (0, 180, 255), (30, 210, 255)])
        cv2.fillPoly(flame_layer, [points], color)
        cv2.circle(flame_layer, (base_x, base_y - 6), max(6, half // 2), color, -1)

    image[:] = cv2.addWeighted(image, 0.30, flame_layer, 0.70, 0)
    return Box(0, x1, y1, x2, y2)


def _draw_smoke(image: np.ndarray, rng: random.Random) -> Box:
    bw = rng.randint(105, 240)
    bh = rng.randint(100, 235)
    x1 = rng.randint(20, WIDTH - bw - 20)
    y1 = rng.randint(18, max(20, HEIGHT - bh - 70))
    x2 = x1 + bw
    y2 = y1 + bh

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for _ in range(rng.randint(12, 24)):
        cx = rng.randint(x1 + 8, x2 - 8)
        cy = rng.randint(y1 + 8, y2 - 8)
        rx = rng.randint(18, max(20, bw // 4))
        ry = rng.randint(16, max(18, bh // 5))
        cv2.ellipse(mask, (cx, cy), (rx, ry), rng.randint(0, 180), 0, 360, rng.randint(110, 220), -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(7, 15))
    alpha = (mask.astype(np.float32) / 255.0 * rng.uniform(0.45, 0.72))[:, :, None]
    smoke_value = rng.randint(70, 205)
    smoke = np.full_like(image, (smoke_value, smoke_value, smoke_value + rng.randint(-8, 8)))
    image[:] = np.clip(image * (1 - alpha) + smoke * alpha, 0, 255).astype(np.uint8)
    return Box(1, x1, y1, x2, y2)


def _draw_negative_distractor(image: np.ndarray, rng: random.Random, mode: str) -> None:
    if mode == "clouds":
        for _ in range(rng.randint(4, 9)):
            x = rng.randint(30, WIDTH - 30)
            y = rng.randint(25, 140)
            r = rng.randint(18, 40)
            cv2.circle(image, (x, y), r, (215, 220, 225), -1)
    elif mode == "fog":
        fog = np.full_like(image, (190, 195, 200))
        image[:] = cv2.addWeighted(image, 0.55, fog, 0.45, 0)
    elif mode == "steam":
        layer = image.copy()
        for i in range(6):
            x = 230 + i * 20 + rng.randint(-8, 8)
            cv2.ellipse(layer, (x, 250 - i * 24), (25, 55), rng.randint(-15, 15), 0, 360, (220, 220, 220), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 7)
        image[:] = cv2.addWeighted(image, 0.7, layer, 0.3, 0)
    elif mode == "extinguished":
        for i in range(7):
            cv2.line(image, (250 + i * 14, 320), (310 + i * 5, 280), (45, 42, 40), 7)
    else:
        for _ in range(12):
            x1 = rng.randint(20, WIDTH - 80)
            y1 = rng.randint(180, HEIGHT - 50)
            cv2.rectangle(image, (x1, y1), (x1 + rng.randint(25, 70), y1 + rng.randint(20, 55)), (70, 85, 100), -1)


def generate_sample(kind: str, index: int, seed: int) -> tuple[np.ndarray, list[Box]]:
    rng = random.Random(seed)
    night = rng.random() < 0.28
    scene = rng.choice(["forest", "residential", "indoor", "camp", "industrial"])
    image = _gradient_background(rng, night)
    _draw_scene(image, rng, scene, night)

    boxes: list[Box] = []
    if kind == "fire":
        boxes.append(_draw_fire(image, rng))
    elif kind == "smoke":
        boxes.append(_draw_smoke(image, rng))
    elif kind == "both":
        fire_box = _draw_fire(image, rng)
        smoke_box = _draw_smoke(image, rng)
        boxes.extend([fire_box, smoke_box])
    elif kind == "negative":
        _draw_negative_distractor(image, rng, rng.choice(["clouds", "fog", "steam", "extinguished", "objects"]))
    else:
        raise ValueError(f"Tipo desconhecido: {kind}")

    # Pequenas variações de captura para simular câmeras diferentes.
    if rng.random() < 0.45:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    gain = rng.uniform(0.88, 1.16)
    beta = rng.randint(-12, 12)
    image = cv2.convertScaleAbs(image, alpha=gain, beta=beta)
    return image, boxes


def _write_sample(dataset: Path, split: str, filename: str, image: np.ndarray, boxes: list[Box]) -> None:
    image_path = dataset / split / "images" / filename
    label_path = dataset / split / "labels" / f"{Path(filename).stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    label_path.write_text("\n".join(box.yolo(WIDTH, HEIGHT) for box in boxes), encoding="utf-8")


def build_dataset(dataset: Path, seed: int = 20260824) -> None:
    for split in ["train", "val", "test"]:
        for folder in ["images", "labels"]:
            path = dataset / split / folder
            path.mkdir(parents=True, exist_ok=True)
            for old in path.glob("*"):
                if old.is_file():
                    old.unlink()

    # 120 imagens: 45 Fire, 45 Smoke, 10 Fire+Smoke, 20 negativas.
    split_plan = {
        "train": {"fire": 31, "smoke": 31, "both": 7, "negative": 15},
        "val": {"fire": 9, "smoke": 9, "both": 2, "negative": 4},
        "test": {"fire": 5, "smoke": 5, "both": 1, "negative": 1},
    }

    global_index = 0
    for split, kinds in split_plan.items():
        for kind, amount in kinds.items():
            for local_index in range(amount):
                global_index += 1
                sample_seed = seed + global_index * 7919
                image, boxes = generate_sample(kind, global_index, sample_seed)
                filename = f"{kind}_{global_index:04d}.jpg"
                _write_sample(dataset, split, filename, image, boxes)

    yaml = (
        "path: datasets/fire_smoke\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: Fire\n"
        "  1: Smoke\n"
    )
    (dataset / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"Dataset criado em {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset sintético anotado Fire/Smoke.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    build_dataset(args.output, args.seed)


if __name__ == "__main__":
    main()
