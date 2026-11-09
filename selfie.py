"""Selfie-style score: is this a "posing for the camera, hoping for compliments" video?

Combines frame signals (framing, pose, expression, image quality) with the caption.
It scores the *style of the video*, not the person's looks, gender or identity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from detector import FrameResult

# Caption phrases typical of selfie / look / compliment-seeking posts. Lowercase.
CAPTION_KEYWORDS = [
    # English
    "selfie", "selca", "ootd", "grwm", "outfit", "makeup", "make up", "nomakeup", "no makeup",
    "beauty", "look", "aesthetic", "glow up", "glowup", "transition", "rate me", "how do i look",
    "pretty", "cute", "filter", "pose", "mirror", "camera roll", "photodump", "photo dump",
    # Vietnamese
    "tự sướng", "xinh", "đẹp", "dễ thương", "mặt mộc", "trang điểm", "outfit hôm nay",
    "đẹp không", "xinh không", "khen", "chấm điểm", "mấy điểm", "lên đồ", "make up",
]
_HASHTAG = re.compile(r"#\w+", re.UNICODE)

WEIGHTS = {"framing": 0.30, "pose": 0.15, "expression": 0.20, "caption": 0.20, "quality": 0.15}


@dataclass
class SelfieScore:
    score: int  # 0..100
    parts: dict[str, float]
    caption_hits: list[str]


def frame_signal(result: FrameResult) -> float:
    """Caption-free part of the score; used to pick the best frame of a video."""
    if not result.face_detected:
        return 0.0
    return (WEIGHTS["framing"] * result.framing + WEIGHTS["pose"] * result.details.get("pose", 0)
            + WEIGHTS["expression"] * expression_signal(result) + WEIGHTS["quality"] * result.quality / 100)


def expression_signal(result: FrameResult) -> float:
    """Posing for the camera = smiling, or a calm/confident neutral face."""
    p = result.expression_probs
    return min(1.0, p.get("happy", 0) + 0.7 * p.get("neutral", 0) + 0.4 * p.get("surprised", 0))


def caption_signal(caption: str, has_effect: bool) -> tuple[float, list[str]]:
    text = caption.lower()
    hits = [kw for kw in CAPTION_KEYWORDS if kw in text]
    score = 0.35 * len(hits)
    if has_effect:  # beauty / face filters are a strong selfie hint
        score += 0.4
        hits.append("[effect filter]")
    if not hits and len(_HASHTAG.findall(text)) == 0 and len(text.strip()) <= 15:
        score += 0.15  # short / emoji-only captions are common on selfie posts
    return min(1.0, score), hits


def score(result: FrameResult, caption: str, has_effect: bool) -> SelfieScore:
    if not result.face_detected:
        return SelfieScore(0, {}, [])
    cap, hits = caption_signal(caption, has_effect)
    parts = {
        "framing": result.framing,
        "pose": result.details.get("pose", 0),
        "expression": expression_signal(result),
        "caption": cap,
        "quality": result.quality / 100,
    }
    total = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()))
    return SelfieScore(total, {k: round(float(v), 2) for k, v in parts.items()}, hits)
