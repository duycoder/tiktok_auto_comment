"""Face detection + image quality scoring for a single frame.

Quality score is about the image only (sharpness, lighting, contrast, face size,
frontal pose). The largest face also gets a gender / age estimate (InsightFace).

CLI (step 1 of the roadmap):  python detector.py shot1.png shot2.jpg ...
"""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
_MODELS_DIR = Path(__file__).parent / "models"
YUNET = "face_detection_yunet/face_detection_yunet_2023mar.onnx"
FER = "facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx"
EXPRESSIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
# 112x112 alignment template for the FER model (same order as YuNet landmarks).
_FER_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
_detector: cv2.FaceDetectorYN | None = None
_fer: cv2.dnn.Net | None = None
# InsightFace genderage (buffalo_l): continuous age + gender, 96x96 crop.
_INSIGHT_URL = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/"
INSIGHT_MODEL = "genderage.onnx"
_insight_net: cv2.dnn.Net | None = None


def _model_path(zoo_path: str, base: str = _ZOO) -> Path:
    path = _MODELS_DIR / Path(zoo_path).name
    if not path.exists():
        _MODELS_DIR.mkdir(exist_ok=True)
        print(f"Downloading model -> {path}")
        urllib.request.urlretrieve(base + zoo_path, path)
    return path


def _get_insight() -> cv2.dnn.Net:
    global _insight_net
    if _insight_net is None:
        _insight_net = cv2.dnn.readNet(str(_model_path(INSIGHT_MODEL, _INSIGHT_URL)))
    return _insight_net


def _get_detector() -> cv2.FaceDetectorYN:
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(_model_path(YUNET)), "", (320, 320), score_threshold=0.7)
    return _detector


def _get_fer() -> cv2.dnn.Net:
    global _fer
    if _fer is None:
        _fer = cv2.dnn.readNet(str(_model_path(FER)))
    return _fer


@dataclass
class FrameResult:
    face_detected: bool
    faces: list[tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h
    quality: int = 0  # 0..100
    details: dict[str, float] = field(default_factory=dict)
    expression: str = ""
    expression_probs: dict[str, float] = field(default_factory=dict)
    framing: float = 0.0  # 0..1, how much the frame looks like a selfie close-up
    female_prob: float = 0.0  # 0..1, of the largest face (InsightFace)
    face_score: float = 0.0  # YuNet confidence of the largest face (0.7..1; low = maybe not a face)
    age_estimate: float = 0.0  # InsightFace estimated age (reads young adults several years older)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _detect(image: np.ndarray, min_side: int) -> np.ndarray:
    """YuNet rows: x, y, w, h, 5 landmarks (eyes, nose, mouth corners) as x/y pairs, score."""
    det = _get_detector()
    det.setInputSize((image.shape[1], image.shape[0]))
    _, faces = det.detect(image)
    if faces is None:
        return np.empty((0, 15), np.float32)
    return faces[(faces[:, 2] >= min_side) & (faces[:, 3] >= min_side)]


def _pose_score(face: np.ndarray) -> float:
    """1.0 = looking at the camera; drops as the nose drifts off the eye midline (yaw) or eyes tilt (roll)."""
    rx, ry, lx, ly, nx, _ = face[4:10]
    eye_dist = max(np.hypot(lx - rx, ly - ry), 1.0)
    yaw = abs(nx - (rx + lx) / 2) / eye_dist  # ~0 frontal, ~0.5 strong profile
    roll = abs(np.degrees(np.arctan2(ly - ry, lx - rx)))
    roll = min(roll, 180 - roll)
    return _clamp01(1.0 - yaw * 1.6) * _clamp01(1.0 - roll / 60.0)


def _expression(image: np.ndarray, face: np.ndarray) -> dict[str, float]:
    """Align the face to the FER template and return expression probabilities."""
    matrix, _ = cv2.estimateAffinePartial2D(face[4:14].reshape(5, 2), _FER_TEMPLATE)
    if matrix is None:
        return {}
    crop = cv2.warpAffine(image, matrix, (112, 112))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis]
    net = _get_fer()
    net.setInput(blob)
    logits = net.forward()[0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return {name: round(float(p), 3) for name, p in zip(EXPRESSIONS, probs)}


def _insight_age_gender(image: np.ndarray, face: np.ndarray) -> tuple[float, float]:
    """(female probability, age) from InsightFace genderage: bbox-centered 1.5x crop, 96x96 RGB."""
    x, y, w, h = face[:4]
    scale = 96 / (max(w, h) * 1.5)
    matrix = np.array([[scale, 0, 48 - (x + w / 2) * scale], [0, scale, 48 - (y + h / 2) * scale]], np.float32)
    crop = cv2.warpAffine(image, matrix, (96, 96))
    net = _get_insight()
    net.setInput(cv2.dnn.blobFromImage(crop, 1.0, (96, 96), (0, 0, 0), swapRB=True))
    out = net.forward()[0]
    logits = out[:2] - out[:2].max()  # [female, male]
    probs = np.exp(logits) / np.exp(logits).sum()
    return float(probs[0]), float(out[2] * 100)


def _framing(image: np.ndarray, faces: np.ndarray, face: np.ndarray) -> float:
    """Selfie-style framing: one big, centered face."""
    img_h, img_w = image.shape[:2]
    x, _, w, h = face[:4]
    closeness = _clamp01((w * h) / (img_w * img_h) / 0.10)
    off_center = abs((x + w / 2) - img_w / 2) / (img_w / 2)
    centered = _clamp01(1.0 - off_center * 0.8)
    crowd = {1: 1.0, 2: 0.6}.get(len(faces), 0.3)
    return closeness * centered * crowd


def analyze(image: np.ndarray) -> FrameResult:
    """Analyze a BGR image."""
    faces = _detect(image, min_side=max(24, min(image.shape[:2]) // 25))
    if not len(faces) and max(image.shape[:2]) < 1400:
        # Small / distant faces: look again on a 2x upscaled copy and keep working on that copy.
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        faces = _detect(image, min_side=max(24, min(image.shape[:2]) // 25))
    if not len(faces):
        return FrameResult(face_detected=False)

    # Score the largest face; that's the subject of the video.
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = (int(v) for v in face[:4])
    x, y = max(x, 0), max(y, 0)
    roi = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    sharpness = _clamp01(cv2.Laplacian(roi, cv2.CV_64F).var() / 300.0)
    brightness = _clamp01(1.0 - abs(roi.mean() - 130.0) / 130.0)
    contrast = _clamp01(roi.std() / 60.0)
    size = _clamp01((w * h) / (image.shape[0] * image.shape[1]) / 0.08)
    pose = _pose_score(face)

    weights = {"sharpness": 0.35, "brightness": 0.2, "contrast": 0.15, "size": 0.15, "pose": 0.15}
    details = {"sharpness": sharpness, "brightness": brightness, "contrast": contrast, "size": size, "pose": pose}
    quality = round(100 * sum(weights[k] * details[k] for k in weights))
    boxes = [tuple(int(v) for v in f[:4]) for f in faces]
    probs = _expression(image, face)
    female, age = _insight_age_gender(image, face)
    return FrameResult(True, boxes, quality, {k: round(float(v), 2) for k, v in details.items()},
                       expression=max(probs, key=probs.get) if probs else "",
                       expression_probs=probs, framing=round(float(_framing(image, faces, face)), 2),
                       female_prob=round(female, 3), age_estimate=round(age, 1),
                       face_score=round(float(face[14]), 3))


def analyze_bytes(data: bytes) -> FrameResult:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode image")
    return analyze(image)


def dhash(image: np.ndarray, size: int = 8) -> int:
    """Perceptual hash used to drop duplicate frames when the URL is unknown."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python detector.py <image> [image ...]")
    for i, path in enumerate(sys.argv[1:], 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{i:02d}] {path}: cannot read")
            continue
        r = analyze(img)
        print(f"[{i:02d}] {path}")
        print(f"     Face detected: {'YES' if r.face_detected else 'NO'}")
        if r.face_detected:
            print(f"     Faces: {len(r.faces)} (score {r.face_score:.2f})  Image quality: {r.quality}/100  {r.details}")
            print(f"     Expression: {r.expression} {r.expression_probs}  Framing: {r.framing}")
            print(f"     Female: {r.female_prob:.0%}  Age ~{r.age_estimate:.0f}")
