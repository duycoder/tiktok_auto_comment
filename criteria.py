"""Match criteria: an adult woman.

Age comes from InsightFace genderage, which reads young adults several years
older than they are (tested: a 21-year-old read as ~32). The estimated-age floor
MIN_AGE_FLOOR (30 estimated ~ 20 real) is a hard bound that keeps real under-18s
out, applied to the YOUNGEST reading over all sampled frames. Captions that
hint at a school-age poster are always skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from detector import FrameResult

MIN_AGE_FLOOR = 30           # hard lower bound for the InsightFace estimated age (~20 real)
DEFAULT_MIN_FEMALE = 0.70
MIN_FACE_SCORE = 0.85        # YuNet confidence needed to treat a detection as a real face
MIN_FACE_AREA = 0.005        # face must cover >= 0.5% of the frame (tiny "faces" are mostly false hits)

# Caption hints that the poster may be a minor / student (Vietnamese + English). Lowercase.
_MINOR_PATTERNS = [
    r"học sinh", r"\bhs\b", r"hocsinh", r"lớp\s?\d{1,2}", r"\blop\s?\d{1,2}", r"cấp\s?[23]", r"cấp ba", r"cấp hai",
    r"\bcap\s?[23]\b", r"thcs", r"thpt", r"\bthpt\b", r"đồng phục", r"dong phuc", r"áo dài trắng", r"đi học",
    r"\b2k(8|9|1\d)\b", r"\b20(0[89]|1\d)\b", r"\b(1[0-7])\s?(t|tuổi|tuoi)\b", r"tuổi\s?(1[0-7])\b",
    r"\bteen\b", r"school", r"\bgrade\s?\d", r"\b(1[0-7])\s?(yo|y/o|years? old)\b", r"sinh năm 20(0[89]|1\d)",
]
_MINOR_RE = re.compile("|".join(_MINOR_PATTERNS), re.IGNORECASE | re.UNICODE)
# Same idea for usernames, which have no word boundaries ("dauthichdoc2010", "linh2k9", "hs_lop10").
_NICK_MINOR_RE = re.compile(
    r"(?<!\d)20(0[89]|1\d)(?!\d)|(?<![a-z\d])2k(8|9|1\d)(?!\d)|hocsinh|(?<![a-z])hs(?![a-z])|lop\d{1,2}|thcs|thpt",
    re.IGNORECASE)


@dataclass
class Verdict:
    match: bool
    reason: str                 # short Vietnamese label shown in the UI
    female: float = 0.0         # best frame's female probability
    age: float = 0.0            # youngest estimated age over all frames
    flags: list[str] = field(default_factory=list)
    # Rejected only on a model guess (age / gender), so the user may override it after
    # looking at the video. Never set for caption minor hints or missing faces.
    overridable: bool = False

    @property
    def label(self) -> str:
        if not self.female and not self.age:
            return "-"
        return f"Nữ {self.female:.0%} · ~{self.age:.0f}t"


def frame_signal(result: FrameResult) -> float:
    """Used to pick the clearest frame of a video: confident face, good image."""
    if not result.face_detected:
        return 0.0
    return result.female_prob * 0.6 + result.quality / 100 * 0.4


def minor_hints(caption: str) -> list[str]:
    return sorted({m.group(0).strip() for m in _MINOR_RE.finditer(caption or "")})


def nick_minor_hints(author: str | None) -> list[str]:
    return sorted({"@" + m.group(0) for m in _NICK_MINOR_RE.finditer(author or "")})


def real_face(result: FrameResult) -> bool:
    """Confident, not-tiny detection (filters hands, patterns, road signs, game HUDs...)."""
    return result.face_score >= MIN_FACE_SCORE and result.details.get("size", 0) * 0.08 >= MIN_FACE_AREA


def evaluate(results: list[FrameResult], best: FrameResult, caption: str,
             min_age: int = MIN_AGE_FLOOR, max_age: int = 99, min_female: float = DEFAULT_MIN_FEMALE,
             author: str | None = None) -> Verdict:
    min_age = max(min_age, MIN_AGE_FLOOR)
    if not best.face_detected:
        return Verdict(False, "không có mặt")
    faces = [r for r in results if r.face_detected] or [best]
    youngest = min(r.age_estimate for r in faces)
    v = Verdict(False, "", female=best.female_prob, age=youngest)

    hints = minor_hints(caption)
    nick_hints = nick_minor_hints(author)
    if hints:
        v.flags = hints
        v.reason = "caption có dấu hiệu học sinh"
    elif nick_hints:
        v.flags = nick_hints
        v.reason = "tên nick có dấu hiệu học sinh / năm sinh"
    elif not real_face(best):
        v.reason = "không chắc có mặt người"
    elif best.female_prob < min_female:
        # Not offered for review: in practice these are men, objects and game clips.
        v.reason = "không phải nữ / không chắc"
    elif v.age < min_age:
        v.reason, v.overridable = f"có thể dưới tuổi (ước tính < {min_age})", True
    elif v.age > max_age:
        v.reason, v.overridable = f"tuổi ước tính > {max_age}", True
    else:
        v.match = True
        v.reason = "PHÙ HỢP"
    return v
