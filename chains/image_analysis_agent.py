from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[1] / "trained_bird_classifier.pt"
yolo_model = YOLO(str(MODEL_PATH))


def classify_bird_image(image_source: str | Path | Any) -> dict:
    """Classify a bird image using the YOLO classifier model."""
    if hasattr(image_source, "read"):
        image = Image.open(image_source).convert("RGB")
        image_path = getattr(image_source, "filename", "uploaded-image")
    else:
        image = Path(image_source)
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        image_path = str(image)

    results = yolo_model(image, stream=False)[0]
    predicted_class = results.names[results.probs.top1]
    confidence = float(results.probs.top1conf)

    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
    }

