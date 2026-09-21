from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.detector import FireSmokeDetector  # noqa: E402


def test_image(detector: FireSmokeDetector, path: Path) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Não foi possível ler a imagem: {path}")
    detections = detector.detect(frame)
    output = detector.draw_detections(frame, detections)
    print(f"Detecções: {len(detections)}")
    for det in detections:
        print(det)
    cv2.imshow("FireWatch AI - imagem", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_stream(detector: FireSmokeDetector, source: str) -> None:
    capture_source: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")

    print("Pressione Q para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        output = detector.draw_detections(frame, detections)
        cv2.imshow("FireWatch AI - teste", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa o modelo em imagem, vídeo ou webcam.")
    parser.add_argument("source", help="Caminho de imagem/vídeo ou índice da webcam, ex.: 0")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "fire_model.pt")
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()

    detector = FireSmokeDetector(args.model, args.confidence)
    if not detector.available:
        raise RuntimeError(detector.error_message or "Modelo indisponível.")

    source_path = Path(args.source)
    if args.source.isdigit():
        test_stream(detector, args.source)
    elif source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        test_image(detector, source_path)
    else:
        test_stream(detector, args.source)


if __name__ == "__main__":
    main()
