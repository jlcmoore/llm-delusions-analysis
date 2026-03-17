"""Shared redaction helpers for the chatlog pipeline package.

This mirrors scripts/redaction_utils.py so that modules under this package
can import it via relative imports without modifying sys.path.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}

SOFT_BREAK_CHARS = "\n\r\t .,!?:;)]}>"

FAKER_ENTITY_MAP = {
    "PERSON": "name",
    "FIRST_NAME": "first_name",
    "LAST_NAME": "last_name",
    "MIDDLE_NAME": "name",
    "LOCATION": "city",
    "CITY": "city",
    "STATE": "state",
    "COUNTRY": "country",
    "ZIP": "postcode",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone_number",
    "IP_ADDRESS": "ipv4",
    "CREDIT_CARD": "credit_card_number",
}


class FakerState:
    """Stateful Faker helper yielding deterministic replacements per entity."""

    def __init__(self, locale: Optional[str] = None):
        self._locale = locale or "en_US"
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], str] = {}
        self._recent_keys: List[Tuple[str, str]] = []

    def replacement(self, entity_type: str, original: str) -> str:
        """Return a deterministic faker replacement for (entity_type, original)."""
        norm_type = (entity_type or "DEFAULT").upper()
        source = original or norm_type
        key = (norm_type, source)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            value = self._generate(norm_type, source)
            self._cache[key] = value
            self._recent_keys.append(key)
            return value

    def _generate(self, entity_type: str, original: str) -> str:
        """Generate a deterministic fake using a hashed seed."""
        seed_bytes = f"{self._locale}|{entity_type}|{original}".encode(
            "utf-8", "ignore"
        )
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
        faker = Faker(self._locale)
        faker.seed_instance(seed)
        provider = FAKER_ENTITY_MAP.get(entity_type, "name")
        generator = getattr(faker, provider, faker.name)
        return generator()

    def consume_new_identifiers(self) -> List[str]:
        """Return newly generated faker values since the last call and clear them."""
        with self._lock:
            values = [self._cache[k] for k in self._recent_keys]
            self._recent_keys.clear()
        return values


def op_params_for(
    operator: str,
    replace_with: str,
    mask_char: str,
    mask_chars_to_mask: int,
    mask_from_end: bool,
) -> Dict:
    """Return operator parameter mapping for the selected anonymizer operator."""
    if operator == "replace":
        return {"new_value": replace_with}
    if operator == "mask":
        return {
            "masking_char": mask_char,
            "chars_to_mask": mask_chars_to_mask,
            "from_end": bool(mask_from_end),
        }
    return {}


def _apply_faker(
    text: str,
    results,
    faker_state: FakerState,
) -> str:
    """Return text with detected spans replaced by Faker-generated values."""
    pieces: List[str] = []
    last = 0
    for res in sorted(results, key=lambda r: (r.start, r.end)):
        start = int(res.start)
        end = int(res.end)
        if start < last:
            continue  # Skip overlaps already handled.
        pieces.append(text[last:start])
        original = text[start:end]
        pieces.append(faker_state.replacement(res.entity_type, original))
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def anonymize_text(
    *,
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    entities: Optional[List[str]],
    score_threshold: float,
    operator: str,
    op_params: Dict,
    allow_list: Optional[List[str]],
    allow_list_match: str,
    faker_state: Optional[FakerState],
) -> str:
    """Anonymize PII in a text string using Presidio."""
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=entities,
        score_threshold=score_threshold,
        allow_list=allow_list,
        allow_list_match=allow_list_match,
    )
    if not results:
        return text
    if operator == "faker":
        if faker_state is None:
            raise RuntimeError("FakerState required when operator='faker'.")
        return _apply_faker(text, results, faker_state)
    unique_types = {r.entity_type for r in results}
    operators = {t: OperatorConfig(operator, op_params) for t in unique_types}
    return anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    ).text


def anonymize_string(
    *,
    name: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    entities: Optional[List[str]],
    score_threshold: float,
    operator: str,
    op_params: Dict,
    allow_list: Optional[List[str]] = None,
    allow_list_match: str = "exact",
    faker_state: Optional[FakerState] = None,
) -> str:
    """Anonymize PII in a short name string (e.g., file or directory name)."""
    results = analyzer.analyze(
        text=name,
        language=language,
        entities=entities,
        score_threshold=score_threshold,
        allow_list=allow_list,
        allow_list_match=allow_list_match,
    )
    if not results:
        return name
    if operator == "faker":
        if faker_state is None:
            raise RuntimeError("FakerState required when operator='faker'.")
        return _apply_faker(name, results, faker_state)
    unique_types = {r.entity_type for r in results}
    operators = {t: OperatorConfig(operator, op_params) for t in unique_types}
    return anonymizer.anonymize(
        text=name, analyzer_results=results, operators=operators
    ).text


def safe_fs_component(
    s: str,
    fallback: str = "REDACTED",
    *,
    faker_state: Optional[FakerState] = None,
    entity_type: str = "PERSON",
) -> str:
    """Return a sanitized filesystem component, safe across platforms.

    Strips and replaces invalid characters, trims trailing spaces/dots, and
    avoids reserved Windows device names. Falls back to either the provided
    fallback string or a Faker-generated value when faker_state is supplied.
    """

    def _sanitize(value: str) -> str:
        cleaned = value.strip()
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", cleaned)
        cleaned = cleaned.rstrip(" .")
        return cleaned[:240] if len(cleaned) > 240 else cleaned

    candidate = _sanitize(s)
    if not candidate:
        fallback_value = (
            fallback
            if faker_state is None
            else faker_state.replacement(entity_type, fallback)
        )
        candidate = _sanitize(fallback_value) or "anon"

    if candidate.upper() in WINDOWS_RESERVED:
        reserved_replacement = (
            f"_{candidate}_"
            if faker_state is None
            else faker_state.replacement(entity_type, f"{candidate}_reserved")
        )
        candidate = _sanitize(reserved_replacement)

    return candidate[:240] if len(candidate) > 240 else candidate


def unique_name_in(parent: Path, base: str, ext: str = "") -> str:
    """Return a unique name in a directory by appending a numeric suffix."""
    candidate = f"{base}{ext}"
    idx = 2
    while (parent / candidate).exists():
        candidate = f"{base}_{idx}{ext}"
        idx += 1
    return candidate


def split_name_and_ext(filename: str) -> Tuple[str, str]:
    """Split filename into stem and combined extension (supports multi-suffix)."""
    p = Path(filename)
    ext = "".join(p.suffixes)
    stem = filename[: -len(ext)] if ext else filename
    return stem, ext


def compute_chunk_spans(
    text: str, max_chars: int = 200_000, soft_break_window: int = 2000
) -> List[Tuple[int, int]]:
    """Compute chunk spans preferring to break near soft separators.

    Returns a list of (start, end) indices that cover the text without overlap.
    """
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        limit = min(i + max_chars, n)
        if limit == n:
            spans.append((i, n))
            break
        win_start = max(i, limit - soft_break_window)
        window = text[win_start:limit]
        last_pos = -1
        for ch in SOFT_BREAK_CHARS:
            pos = window.rfind(ch)
            last_pos = max(last_pos, pos)
        cut = win_start + last_pos + 1 if last_pos != -1 else limit
        spans.append((i, cut))
        i = cut
    return spans


def anonymize_chunked_text(
    *,
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    entities: Optional[List[str]],
    score_threshold: float,
    operator: str,
    op_params: Dict,
    allow_list: Optional[List[str]],
    allow_list_match: str,
    faker_state: Optional[FakerState],
    chunk_size: int,
    chunk_break_window: int,
) -> str:
    """Anonymize text, falling back to chunked processing for large inputs.

    Uses compute_chunk_spans to split long texts into manageable chunks while
    preserving word boundaries where possible. Each chunk is independently
    anonymized via anonymize_text, and the results are concatenated.

    Parameters:
    - text: Input string to anonymize.
    - analyzer: Presidio AnalyzerEngine instance.
    - anonymizer: Presidio AnonymizerEngine instance.
    - language: Language code for Presidio.
    - entities: Optional list of entity types to detect.
    - score_threshold: Minimum confidence score for detected entities.
    - operator: Presidio anonymizer operator name.
    - op_params: Operator-specific parameters.
    - allow_list: Optional list of terms to exclude from detection.
    - allow_list_match: Matching strategy for allow_list.
    - faker_state: Optional FakerState for faker-based replacements.
    - chunk_size: Maximum characters per chunk before splitting.
    - chunk_break_window: Window size used to find soft breakpoints.

    Returns:
    - The anonymized text, with the same length segmentation semantics as if
      anonymize_text had been applied to the whole string.
    """
    if len(text) <= chunk_size:
        return anonymize_text(
            text=text,
            analyzer=analyzer,
            anonymizer=anonymizer,
            language=language,
            entities=entities,
            score_threshold=score_threshold,
            operator=operator,
            op_params=op_params,
            allow_list=allow_list,
            allow_list_match=allow_list_match,
            faker_state=faker_state,
        )

    spans = compute_chunk_spans(
        text, max_chars=chunk_size, soft_break_window=chunk_break_window
    )
    parts: List[str] = []
    for start, end in spans:
        part = text[start:end]
        parts.append(
            anonymize_text(
                text=part,
                analyzer=analyzer,
                anonymizer=anonymizer,
                language=language,
                entities=entities,
                score_threshold=score_threshold,
                operator=operator,
                op_params=op_params,
                allow_list=allow_list,
                allow_list_match=allow_list_match,
                faker_state=faker_state,
            )
        )
    return "".join(parts)


def is_text_file(p: Path, include_all: bool) -> bool:
    """Heuristically decide if a path should be treated as text.

    If include_all is True, treats any file without NUL bytes in the first
    2048 bytes as text. Otherwise returns False (caller can use extension sets).
    """
    if include_all:
        try:
            with p.open("rb") as f:
                chunk = f.read(2048)
            return b"\x00" not in chunk
        except OSError:
            return False
    return False


def read_text(p: Path) -> Tuple[str, str]:
    """Read file contents as text with common encoding fallbacks.

    Tries UTF-8, UTF-8 with BOM, and Latin-1 before falling back to UTF-8 with
    replacement. Returns (text, encoding).
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="ignore"), "utf-8"
