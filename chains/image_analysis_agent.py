from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[1] / "trained_bird_classifier.pt"
yolo_model = YOLO(str(MODEL_PATH))


def classify_bird_image(image_path: str | Path) -> dict:
    """Classify a bird image using the YOLO classifier model."""
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    results = yolo_model(image, stream=False)[0]
    predicted_class = results.names[results.probs.top1]
    confidence = float(results.probs.top1conf)

    return {
        "image_path": str(image),
        "predicted_class": predicted_class,
        "confidence": confidence,
    }

