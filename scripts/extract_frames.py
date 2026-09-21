from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai frames de um vídeo para criação/expansão do dataset.")
    parser.add_argument("video", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--every-frames", type=int, default=30, help="Extrai um frame a cada N frames.")
    group.add_argument("--every-seconds", type=float, help="Extrai um frame a cada N segundos.")
    parser.add_argument("--output", type=Path, default=Path("reports/extracted_frames"))
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("OpenCV não conseguiu abrir o vídeo.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if args.every_seconds is not None:
        if fps <= 0:
            raise RuntimeError("FPS inválido; não é possível usar --every-seconds.")
        interval = max(1, round(fps * args.every_seconds))
    else:
        interval = max(1, int(args.every_frames))

    args.output.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % interval == 0:
            path = args.output / f"frame_{frame_index:07d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
        frame_index += 1

    cap.release()
    print(f"{saved} frames salvos em {args.output}")


if __name__ == "__main__":
    main()
