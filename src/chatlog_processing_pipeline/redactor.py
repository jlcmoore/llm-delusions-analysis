"""Anonymization helpers for content and file names using Presidio.

This module mirrors an input directory structure under an output directory and
applies PII anonymization to message content in parsed JSONs as well as to raw
text files when needed. Behavior unchanged; comments and docstring only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import en_core_web_lg
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm

from .redaction_utils import FakerState
from .redaction_utils import anonymize_chunked_text as util_anonymize_chunked_text
from .redaction_utils import anonymize_string as util_anonymize_string
from .redaction_utils import anonymize_text as util_anonymize_text
from .redaction_utils import compute_chunk_spans as util_compute_chunk_spans
from .redaction_utils import is_text_file, op_params_for
from .redaction_utils import read_text as util_read_text
from .redaction_utils import safe_fs_component, split_name_and_ext, unique_name_in
from .util import (  # safe JSON writer with surrogate cleanup
    iter_chatgpt_messages,
    looks_like_chatgpt_mapping,
    looks_like_parsed_chat_json,
    normalize_meta_dict,
    write_json,
)

# Text-like extensions we treat as text by default (without --include-all)
TEXT_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".log",
    ".rtf",
    ".tex",
    ".srt",
    ".vtt",
}

PATH_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{2,6}$")

_ANALYZER_CACHE: Dict[Tuple[str, Optional[int], bool], AnalyzerEngine] = {}


def _strip_diacritics(s: str) -> str:
    """Return a copy of string with diacritics removed (NFKD fold)."""
    norm = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _load_contact_identifiers(meta_csv: Path, verbose: bool = False) -> List[str]:
    """Load contact identifiers from a metadata CSV.

    Scans any column whose header contains 'contact' or 'identifier'
    (case-insensitive).
    Extracts emails and adds full phrases and individual tokens, including
    diacritic-stripped variants to improve matching robustness.

    Parameters:
    - meta_csv: Path to transcripts/metadata.csv
    - verbose: Print an info line with the count if True.

    Returns:
    - List of identifier strings to treat as blocklist terms.
    """
    terms: List[str] = []
    if not meta_csv.exists():
        return terms
    try:
        with meta_csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = [c or "" for c in (reader.fieldnames or [])]
            contact_cols = [
                c
                for c in cols
                if any(
                    token in c.strip().lower() for token in ("contact", "identifier")
                )
            ]
            if not contact_cols:
                return terms
            for row in reader:
                for col in contact_cols:
                    cell = (row.get(col) or "").strip()
                    if not cell:
                        continue
                    parts = re.split(r"[;,\n]+", cell)
                    for part in parts:
                        p = part.strip()
                        if not p:
                            continue
                        for m in re.findall(
                            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", p
                        ):
                            for variant in (m, _strip_diacritics(m)):
                                if variant and variant not in terms:
                                    terms.append(variant)
                        # Consider phrases with at least two alpha chars
                        if len(re.findall(r"[A-Za-z\u00C0-\u024F]", p)) >= 2:
                            for variant in (p, _strip_diacritics(p)):
                                if variant and variant not in terms:
                                    terms.append(variant)
                            for tok in p.split():
                                for variant in (tok, _strip_diacritics(tok)):
                                    if variant and variant not in terms:
                                        terms.append(variant)
        if verbose:
            print(f"[INFO] Loaded {len(terms)} contact identifiers from {meta_csv}")
    except (OSError, UnicodeError, csv.Error) as exc:
        print(
            f"[WARN] Failed to read contact identifiers from {meta_csv}: {exc}",
            file=sys.stderr,
        )
    return terms


def _anonymize_pathish(
    *,
    path_text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    name_entities: Optional[List[str]],
    name_threshold: float,
    name_operator: str,
    name_op_params: Dict,
    name_allow_list: Optional[List[str]],
    name_allow_list_match: str,
    name_faker_state: Optional[FakerState],
) -> str:
    """Anonymize a path-like string component-by-component, preserving '/'."""
    parts = path_text.split("/")
    out: List[str] = []
    for comp in parts:
        if not comp:
            out.append(comp)
            continue
        new_comp = util_anonymize_string(
            name=comp,
            analyzer=analyzer,
            anonymizer=anonymizer,
            language=language,
            entities=name_entities,
            score_threshold=name_threshold,
            operator=name_operator,
            op_params=name_op_params,
            allow_list=name_allow_list,
            allow_list_match=name_allow_list_match,
            faker_state=name_faker_state,
        )
        out.append(new_comp)
    return "/".join(out)


def _anonymize_tokenwise(
    *,
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    name_entities: Optional[List[str]],
    name_threshold: float,
    name_operator: str,
    name_op_params: Dict,
    name_allow_list: Optional[List[str]],
    name_allow_list_match: str,
    name_faker_state: Optional[FakerState],
) -> str:
    """Anonymize only alphabetic runs within a string, preserving separators.

    Helps catch names embedded in filenames like "Alex_2-report".
    """
    buf: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isalpha():
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            token = text[i:j]
            new_token = util_anonymize_string(
                name=token,
                analyzer=analyzer,
                anonymizer=anonymizer,
                language=language,
                entities=name_entities,
                score_threshold=name_threshold,
                operator=name_operator,
                op_params=name_op_params,
                allow_list=name_allow_list,
                allow_list_match=name_allow_list_match,
                faker_state=name_faker_state,
            )
            buf.append(new_token)
            i = j
        else:
            buf.append(text[i])
            i += 1
    return "".join(buf)


def _collect_fake_identifiers(
    *,
    content_faker_state: Optional[FakerState],
    name_faker_state: Optional[FakerState],
    operator: str,
    content_op_params: Dict,
    name_operator: str,
    name_op_params: Dict,
) -> List[str]:
    """Return faker-generated identifiers for the current file with sensible fallbacks."""
    identifiers: List[str] = []
    seen: set[str] = set()

    for state in (content_faker_state, name_faker_state):
        if state is None:
            continue
        for value in state.consume_new_identifiers():
            if value not in seen:
                identifiers.append(value)
                seen.add(value)

    if identifiers:
        return identifiers

    fallback_candidates: List[str] = []
    if operator == "replace":
        new_value = content_op_params.get("new_value")
        if isinstance(new_value, str):
            fallback_candidates.append(new_value)
    if name_operator == "replace":
        new_value = name_op_params.get("new_value")
        if isinstance(new_value, str):
            fallback_candidates.append(new_value)
    if not fallback_candidates:
        fallback_candidates.append("REDACTED")

    for value in fallback_candidates:
        if value not in seen:
            identifiers.append(value)
            seen.add(value)
    return identifiers


def _attach_fake_identifiers(obj, identifiers: List[str]) -> None:
    """Attach the fake identifier list to the JSON object, preferring meta when present."""
    if not identifiers:
        return
    if isinstance(obj, dict):
        meta = obj.get("meta")
        if isinstance(meta, dict):
            meta["fake_identifiers"] = identifiers
        else:
            obj["fake_identifiers"] = identifiers


def _build_redaction_params_from_task(
    task: Dict[str, object],
) -> Dict[str, object]:
    """Return keyword arguments for :func:`_redact_single_file` based on a task."""
    src = Path(str(task["src"]))
    dst = Path(str(task["dst"]))
    lang = str(task["lang"])
    spacy_max_length = task.get("spacy_max_length")
    faker_locale = task.get("faker_locale")
    content_use_faker = bool(task.get("content_use_faker", False))
    name_use_faker = bool(task.get("name_use_faker", False))
    verbose = bool(task.get("verbose", False))
    blocklist_terms = task.get("blocklist_terms")
    extra_terms: Optional[List[str]]
    if isinstance(blocklist_terms, list):
        extra_terms = [str(t) for t in blocklist_terms if str(t).strip()]
    else:
        extra_terms = None
    analyzer = _build_analyzer(
        lang, spacy_max_length if isinstance(spacy_max_length, int) else None
    )
    locale = (
        str(faker_locale)
        if isinstance(faker_locale, str) and faker_locale.strip()
        else None
    )
    anonymizer = AnonymizerEngine()
    content_faker_state = FakerState(locale) if content_use_faker else None
    name_faker_state = FakerState(locale) if name_use_faker else None
    params: Dict[str, object] = {
        "src": src,
        "dst": dst,
        "analyzer": analyzer,
        "anonymizer": anonymizer,
        "lang": lang,
        "entities": task.get("entities"),
        "score_threshold": float(task["score_threshold"]),
        "operator": str(task["operator"]),
        "content_op_params": task["content_op_params"],
        "allow_list": task.get("allow_list"),
        "allow_list_match": str(task["allow_list_match"]),
        "chunk_size": int(task["chunk_size"]),
        "chunk_break_window": int(task["chunk_break_window"]),
        "include_all": bool(task["include_all"]),
        "skip_nontext": bool(task["skip_nontext"]),
        "names_only": bool(task["names_only"]),
        "generic_json_strings": bool(task["generic_json_strings"]),
        "dry_run": bool(task.get("dry_run", False)),
        "verbose": verbose,
        "no_progress": bool(task.get("no_progress", True)),
        "content_faker_state": content_faker_state,
        "name_entities": task.get("name_entities"),
        "name_threshold": float(task["name_threshold"]),
        "name_operator": str(task["name_operator"]),
        "name_op_params": task["name_op_params"],
        "name_allow_list": task.get("name_allow_list"),
        "name_allow_list_match": str(task["name_allow_list_match"]),
        "name_faker_state": name_faker_state,
        "blocklist_terms": extra_terms,
    }
    return params


def _process_file_task(task: Dict[str, object]) -> Tuple[int, int, int]:
    """Worker: redact a single file. Returns (text_processed, bin_copied, failed)."""
    try:
        params = _build_redaction_params_from_task(task)
        return _redact_single_file(**params)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError):
        return (0, 0, 1)


def _destination_for_file(
    *,
    src: Path,
    parent_dst: Path,
    content_only: bool,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    name_entities: Optional[List[str]],
    name_threshold: float,
    name_operator: str,
    name_op_params: Dict,
    name_allow_list: Optional[List[str]],
    name_allow_list_match: str,
    name_faker_state: Optional[FakerState],
    overwrite: bool,
) -> Path:
    """Compute the anonymized destination path for a source file."""
    stem, ext = split_name_and_ext(src.name)
    if content_only:
        anon_stem = stem
    else:
        anon_stem = _anonymize_tokenwise(
            text=stem,
            analyzer=analyzer,
            anonymizer=anonymizer,
            language=language,
            name_entities=name_entities,
            name_threshold=name_threshold,
            name_operator=name_operator,
            name_op_params=name_op_params,
            name_allow_list=name_allow_list,
            name_allow_list_match=name_allow_list_match,
            name_faker_state=name_faker_state,
        )
    stem_fallback = src.name if name_faker_state is not None else "file"
    anon_stem = safe_fs_component(
        anon_stem,
        fallback=stem_fallback,
        faker_state=name_faker_state,
    )
    if overwrite:
        anon_name = unique_name_in(parent_dst, anon_stem, ext)
        return parent_dst / anon_name
    return parent_dst / f"{anon_stem}{ext}"


def _redact_single_file(
    *,
    src: Path,
    dst: Path,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    lang: str,
    entities: Optional[List[str]],
    score_threshold: float,
    operator: str,
    content_op_params: Dict,
    allow_list: Optional[List[str]],
    allow_list_match: str,
    chunk_size: int,
    chunk_break_window: int,
    include_all: bool,
    skip_nontext: bool,
    names_only: bool,
    generic_json_strings: bool,
    dry_run: bool,
    verbose: bool,
    no_progress: bool,
    content_faker_state: Optional[FakerState],
    name_entities: Optional[List[str]],
    name_threshold: float,
    name_operator: str,
    name_op_params: Dict,
    name_allow_list: Optional[List[str]],
    name_allow_list_match: str,
    name_faker_state: Optional[FakerState],
    blocklist_terms: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    """Apply redaction to a single file and report the outcome counters."""

    def _handle_nontext() -> Tuple[int, int, int]:
        if skip_nontext:
            if verbose or dry_run:
                print(f"[SKIP] {src} (non-text)")
            return (0, 0, 0)
        if dry_run:
            print(f"[BIN]  {src} -> {dst}")
            return (0, 1, 0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return (0, 1, 0)

    def _copy_names_only() -> Tuple[int, int, int]:
        if dry_run:
            print(f"[TXT-NAMES-ONLY] {src} -> {dst}")
            return (1, 0, 0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return (1, 0, 0)

    def _process_json_file() -> Tuple[int, int, int]:
        if names_only:
            return _copy_names_only()
        try:
            raw_text, _enc = util_read_text(src)
            obj = json.loads(raw_text)
        except json.JSONDecodeError:
            obj = None

        file_key_tokens = ("file", "filename", "filepath", "path", "pointer", "url")

        def _apply_blocklist(text: str) -> str:
            terms = [t for t in (blocklist_terms or []) if t]
            if not terms:
                return text
            # Build two passes: phrases (with whitespace) then single tokens
            phrases = [t for t in terms if any(ch.isspace() for ch in t)]
            tokens = [t for t in terms if not any(ch.isspace() for ch in t)]

            def _replace_func(match: re.Match) -> str:
                val = match.group(0)
                if operator == "faker":
                    if content_faker_state is None:
                        return "REDACTED"
                    return content_faker_state.replacement("PERSON", val)
                if operator == "replace":
                    return str(content_op_params.get("new_value", "<REDACTED>"))
                if operator == "mask":
                    ch = str(content_op_params.get("masking_char", "*"))
                    return ch * len(val)
                if operator == "hash":
                    return hashlib.sha256(val.encode("utf-8", "ignore")).hexdigest()
                return "<REDACTED>"

            out = text
            if phrases:
                # Match whole phrases at token boundaries; allow flexible whitespace
                def _flex(p: str) -> str:
                    return re.escape(p).replace(r"\ ", r"\s+")

                alt = "|".join(
                    _flex(p) for p in sorted(set(phrases), key=len, reverse=True)
                )
                rx = re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)
                out = rx.sub(_replace_func, out)
            if tokens:
                # Single-word tokens at word boundaries
                alt = "|".join(
                    re.escape(t) for t in sorted(set(tokens), key=len, reverse=True)
                )
                rx2 = re.compile(
                    rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])", re.IGNORECASE
                )
                out = rx2.sub(_replace_func, out)
            return out

        def _anonymize_text_value(text: str) -> str:
            pre = _apply_blocklist(text)
            mid = util_anonymize_chunked_text(
                text=pre,
                analyzer=analyzer,
                anonymizer=anonymizer,
                language=lang,
                entities=entities,
                score_threshold=score_threshold,
                operator=operator,
                op_params=content_op_params,
                allow_list=allow_list,
                allow_list_match=allow_list_match,
                faker_state=content_faker_state,
                chunk_size=chunk_size,
                chunk_break_window=chunk_break_window,
            )
            return _apply_blocklist(mid)

        def _anonymize_string_fields(node) -> bool:
            """Recursively anonymize string values in a JSON-like node."""
            changed_local = False
            if isinstance(node, dict):
                for key, value in list(node.items()):
                    key_lower = str(key).lower()
                    if _is_structural_key(key_lower):
                        continue
                    if isinstance(value, str):
                        if _looks_like_path_or_url(value):
                            new_value = _sanitize_fileish_string(
                                value,
                                text_anonymizer=_anonymize_text_value,
                                operator=operator,
                                content_op_params=content_op_params,
                                name_operator=name_operator,
                                name_op_params=name_op_params,
                                name_faker_state=name_faker_state,
                                content_faker_state=content_faker_state,
                            )
                        else:
                            new_value = _anonymize_text_value(value)
                        if new_value != value:
                            node[key] = new_value
                            changed_local = True
                    elif isinstance(value, (dict, list)):
                        if _anonymize_string_fields(value):
                            changed_local = True
                return changed_local
            if isinstance(node, list):
                for idx, item in enumerate(list(node)):
                    if isinstance(item, str):
                        if _looks_like_path_or_url(item):
                            new_item = _sanitize_fileish_string(
                                item,
                                text_anonymizer=_anonymize_text_value,
                                operator=operator,
                                content_op_params=content_op_params,
                                name_operator=name_operator,
                                name_op_params=name_op_params,
                                name_faker_state=name_faker_state,
                                content_faker_state=content_faker_state,
                            )
                        else:
                            new_item = _anonymize_text_value(item)
                        if new_item != item:
                            node[idx] = new_item
                            changed_local = True
                    elif isinstance(item, (dict, list)):
                        if _anonymize_string_fields(item):
                            changed_local = True
                return changed_local
            return False

        def _meta_dict_from(ref) -> Optional[Dict]:
            meta = ref.get("meta") if isinstance(ref, dict) else None
            return meta if isinstance(meta, dict) else None

        def _conversations_list_from(ref) -> Optional[List]:
            if isinstance(ref, dict):
                convs = ref.get("conversations")
                return convs if isinstance(convs, list) else None
            if isinstance(ref, list):
                # Raw ChatGPT exports often use a top-level list of conversations.
                if all(isinstance(item, dict) for item in ref):
                    return ref
            return None

        def _standardize_meta(meta_dict: Dict) -> Dict:
            """Return a meta dict aligned with processor normalization keys.

            Parameters:
            - meta_dict: Raw ``meta`` dictionary loaded from a JSON file.

            Returns:
            - Dictionary including canonical keys from
              :func:`normalize_meta_dict` when a backing ``ParseMeta``-like
              structure was used during creation. Existing keys not covered by
              the normalization helper are preserved as-is.
            """

            # If the required attributes are present, rebuild via normalize_meta_dict
            src_path = meta_dict.get("full_path") or meta_dict.get("src_path")
            rel_path = meta_dict.get("rel_path")
            file_ext = meta_dict.get("file_ext")
            source_guess = meta_dict.get("source_guess")
            message_count = meta_dict.get("message_count", 0)
            if (
                isinstance(src_path, str)
                and isinstance(rel_path, str)
                and isinstance(file_ext, str)
                and isinstance(source_guess, str)
            ):
                temp_meta = type("MetaLike", (), {})()
                temp_meta.src_path = src_path
                temp_meta.rel_path = rel_path
                temp_meta.file_ext = file_ext
                temp_meta.source_guess = source_guess
                temp_meta.message_count = int(message_count) if message_count else 0
                normalized = normalize_meta_dict(temp_meta)
                merged = dict(meta_dict)
                merged.update(normalized)
                return merged
            return meta_dict

        def _anonymize_meta_and_titles(obj_ref) -> None:
            meta = _meta_dict_from(obj_ref)
            if isinstance(meta, dict):
                updated_meta = _standardize_meta(meta)
                obj_ref["meta"] = updated_meta
                meta = updated_meta
                if isinstance(meta.get("rel_path"), str):
                    meta["rel_path"] = _anonymize_pathish(
                        path_text=_apply_blocklist(meta["rel_path"]),
                        analyzer=analyzer,
                        anonymizer=anonymizer,
                        language=lang,
                        name_entities=name_entities,
                        name_threshold=name_threshold,
                        name_operator=name_operator,
                        name_op_params=name_op_params,
                        name_allow_list=name_allow_list,
                        name_allow_list_match=name_allow_list_match,
                        name_faker_state=name_faker_state,
                    )
                if isinstance(meta.get("filename"), str):
                    meta["filename"] = util_anonymize_string(
                        name=_apply_blocklist(meta["filename"]),
                        analyzer=analyzer,
                        anonymizer=anonymizer,
                        language=lang,
                        entities=name_entities,
                        score_threshold=name_threshold,
                        operator=name_operator,
                        op_params=name_op_params,
                        allow_list=name_allow_list,
                        allow_list_match=name_allow_list_match,
                        faker_state=name_faker_state,
                    )
                if isinstance(meta.get("full_path"), str):
                    meta["full_path"] = _anonymize_pathish(
                        path_text=_apply_blocklist(meta["full_path"]),
                        analyzer=analyzer,
                        anonymizer=anonymizer,
                        language=lang,
                        name_entities=name_entities,
                        name_threshold=name_threshold,
                        name_operator=name_operator,
                        name_op_params=name_op_params,
                        name_allow_list=name_allow_list,
                        name_allow_list_match=name_allow_list_match,
                        name_faker_state=name_faker_state,
                    )

            conversations = _conversations_list_from(obj_ref)
            if conversations is None:
                return
            for conv in conversations:
                if not isinstance(conv, dict):
                    continue
                if isinstance(conv.get("title"), str):
                    conv["title"] = util_anonymize_string(
                        name=conv["title"],
                        analyzer=analyzer,
                        anonymizer=anonymizer,
                        language=lang,
                        entities=name_entities,
                        score_threshold=name_threshold,
                        operator=name_operator,
                        op_params=name_op_params,
                        allow_list=name_allow_list,
                        allow_list_match=name_allow_list_match,
                        faker_state=name_faker_state,
                    )
                _anonymize_url_lists(conv)

        def _anonymize_url_lists(conv_dict: Dict) -> None:
            for key in ("safe_urls", "blocked_urls"):
                urls = conv_dict.get(key)
                if not isinstance(urls, list):
                    continue
                new_urls: List[str] = []
                urls_changed = False
                for url in urls:
                    new_url = url
                    if isinstance(url, str):
                        new_url = _anonymize_text_value(url)
                    if new_url != url:
                        urls_changed = True
                    new_urls.append(new_url)
                if urls_changed:
                    conv_dict[key] = new_urls

        def _anonymize_messages(msgs: List[Dict[str, str]]) -> int:
            local_changed = 0
            for message in msgs:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                new_content = _anonymize_text_value(content)
                if new_content != content:
                    message["content"] = new_content
                    local_changed += 1
            return local_changed

        def _anonymize_chatgpt_message(message: Dict) -> int:
            """Anonymize whitelisted fields in a ChatGPT mapping-style message."""
            local_changed = 0

            author = message.get("author")
            if isinstance(author, dict) and isinstance(author.get("name"), str):
                pre_name = _apply_blocklist(author["name"])
                new_name = util_anonymize_string(
                    name=pre_name,
                    analyzer=analyzer,
                    anonymizer=anonymizer,
                    language=lang,
                    entities=name_entities,
                    score_threshold=name_threshold,
                    operator=name_operator,
                    op_params=name_op_params,
                    allow_list=name_allow_list,
                    allow_list_match=name_allow_list_match,
                    faker_state=name_faker_state,
                )
                if new_name != author["name"]:
                    author["name"] = new_name
                    local_changed += 1

            content = message.get("content")
            if isinstance(content, dict):
                text_value = content.get("text")
                if isinstance(text_value, str):
                    updated = _anonymize_text_value(text_value)
                    if updated != text_value:
                        content["text"] = updated
                        local_changed += 1
                parts = content.get("parts")
                parts_changed = False
                if isinstance(parts, list):
                    new_parts: List = []
                    for part in parts:
                        if isinstance(part, str):
                            new_part = _anonymize_text_value(part)
                            if new_part != part:
                                parts_changed = True
                            new_parts.append(new_part)
                        elif isinstance(part, dict):
                            if _anonymize_fileish_fields(
                                part,
                                file_key_tokens=file_key_tokens,
                                text_anonymizer=_anonymize_text_value,
                                operator=operator,
                                content_op_params=content_op_params,
                                name_operator=name_operator,
                                name_op_params=name_op_params,
                                name_faker_state=name_faker_state,
                                content_faker_state=content_faker_state,
                            ):
                                parts_changed = True
                            if _anonymize_string_fields(part):
                                parts_changed = True
                            new_parts.append(part)
                        else:
                            new_parts.append(part)
                    if parts_changed:
                        content["parts"] = new_parts
                        local_changed += 1

                # Explicit user-provided context fields under content
                for ckey in ("user_profile", "user_instructions"):
                    cvalue = content.get(ckey)
                    if isinstance(cvalue, str):
                        updated = _anonymize_text_value(cvalue)
                        if updated != cvalue:
                            content[ckey] = updated
                            local_changed += 1

            metadata = message.get("metadata")
            if isinstance(metadata, dict):
                user_ctx = metadata.get("user_context_message_data")
                if isinstance(user_ctx, dict):
                    for mkey in ("about_user_message", "about_model_message"):
                        mvalue = user_ctx.get(mkey)
                        if isinstance(mvalue, str):
                            updated = _anonymize_text_value(mvalue)
                            if updated != mvalue:
                                user_ctx[mkey] = updated
                                local_changed += 1

                if _anonymize_fileish_fields(
                    metadata,
                    file_key_tokens=file_key_tokens,
                    text_anonymizer=_anonymize_text_value,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                ):
                    local_changed += 1
                if _anonymize_string_fields(metadata):
                    local_changed += 1

            attachments = message.get("attachments")
            if isinstance(attachments, (dict, list)):
                if _anonymize_fileish_fields(
                    attachments,
                    file_key_tokens=file_key_tokens,
                    text_anonymizer=_anonymize_text_value,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                ):
                    local_changed += 1

            files_field = message.get("files")
            if isinstance(files_field, (dict, list)):
                if _anonymize_fileish_fields(
                    files_field,
                    file_key_tokens=file_key_tokens,
                    text_anonymizer=_anonymize_text_value,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                ):
                    local_changed += 1

            return local_changed

        if looks_like_parsed_chat_json(obj):
            if verbose:
                print("  - parsed JSON detected (content-only redaction)")

            changed = 0
            if isinstance(obj.get("messages"), list):
                changed += _anonymize_messages(obj["messages"])
            if isinstance(obj.get("conversations"), list):
                for conv in obj["conversations"]:
                    msgs = conv.get("messages") if isinstance(conv, dict) else None
                    if isinstance(msgs, list):
                        changed += _anonymize_messages(msgs)

            _anonymize_meta_and_titles(obj)

            fake_identifiers = _collect_fake_identifiers(
                content_faker_state=content_faker_state,
                name_faker_state=name_faker_state,
                operator=operator,
                content_op_params=content_op_params,
                name_operator=name_operator,
                name_op_params=name_op_params,
            )
            _attach_fake_identifiers(obj, fake_identifiers)

            if dry_run:
                print(f"[JSON] {src} -> {dst} (changed {changed} message contents)")
            else:
                write_json(dst, obj)
            return (1, 0, 0)

        if looks_like_chatgpt_mapping(obj):
            if verbose:
                print("  - chat mapping JSON detected (whitelist redaction)")

            changed = 0
            conversations = obj.get("conversations") if isinstance(obj, dict) else obj
            if isinstance(conversations, list):
                for message in iter_chatgpt_messages(conversations):
                    if isinstance(message, dict):
                        changed += _anonymize_chatgpt_message(message)

            _anonymize_meta_and_titles(obj)

            fake_identifiers = _collect_fake_identifiers(
                content_faker_state=content_faker_state,
                name_faker_state=name_faker_state,
                operator=operator,
                content_op_params=content_op_params,
                name_operator=name_operator,
                name_op_params=name_op_params,
            )
            _attach_fake_identifiers(obj, fake_identifiers)

            if dry_run:
                print(
                    f"[JSON-MAPPING] {src} -> {dst} (changed {changed} message contents)"
                )
            else:
                write_json(dst, obj)
            return (1, 0, 0)

        if isinstance(obj, (dict, list)) and generic_json_strings:
            if verbose:
                print("  - generic JSON: anonymizing string values only")
            _ = _anonymize_json_strings(
                obj,
                analyzer=analyzer,
                anonymizer=anonymizer,
                language=lang,
                entities=entities,
                score_threshold=score_threshold,
                operator=operator,
                op_params=content_op_params,
                allow_list=allow_list,
                allow_list_match=allow_list_match,
                chunk_size=chunk_size,
                chunk_break_window=chunk_break_window,
                faker_state=content_faker_state,
            )
            fake_identifiers = _collect_fake_identifiers(
                content_faker_state=content_faker_state,
                name_faker_state=name_faker_state,
                operator=operator,
                content_op_params=content_op_params,
                name_operator=name_operator,
                name_op_params=name_op_params,
            )
            _attach_fake_identifiers(obj, fake_identifiers)
            if dry_run:
                print(f"[JSON-GENERIC] {src} -> {dst}")
            else:
                write_json(dst, obj)
            return (1, 0, 0)

        if dry_run:
            print(f"[JSON-PASS] {src} -> {dst}")
            return (1, 0, 0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return (1, 0, 0)

    def _process_text_file() -> Tuple[int, int, int]:
        if names_only:
            return _copy_names_only()
        text, enc = util_read_text(src)
        n_chars = len(text)
        if n_chars <= chunk_size:
            if verbose:
                print(f"  - single chunk ({n_chars} chars)")
            redacted = util_anonymize_text(
                text=text,
                analyzer=analyzer,
                anonymizer=anonymizer,
                language=lang,
                entities=entities,
                score_threshold=score_threshold,
                operator=operator,
                op_params=content_op_params,
                allow_list=allow_list,
                allow_list_match=allow_list_match,
                faker_state=content_faker_state,
            )
        else:
            spans = util_compute_chunk_spans(
                text, max_chars=chunk_size, soft_break_window=chunk_break_window
            )
            if verbose:
                print(
                    "  - chunking",
                    f"({n_chars} chars)",
                    f"into {len(spans)} chunks",
                    f"(max {chunk_size})",
                )
            chunk_iter = enumerate(spans, start=1)
            if not no_progress:
                chunk_iter = tqdm(
                    enumerate(spans, start=1),
                    total=len(spans),
                    desc=f"Chunks: {src.name}",
                    unit="chunk",
                    leave=False,
                )
            parts: List[str] = []
            for cidx, (start, end) in chunk_iter:
                chunk = text[start:end]
                if verbose:
                    print(
                        f"    - chunk {cidx}/{len(spans)}:",
                        f"{start}-{end}",
                        f"({end-start} chars)",
                    )
                parts.append(
                    util_anonymize_text(
                        text=chunk,
                        analyzer=analyzer,
                        anonymizer=anonymizer,
                        language=lang,
                        entities=entities,
                        score_threshold=score_threshold,
                        operator=operator,
                        op_params=content_op_params,
                        allow_list=allow_list,
                        allow_list_match=allow_list_match,
                        faker_state=content_faker_state,
                    )
                )
            redacted = "".join(parts)

        if dry_run:
            print(f"[TXT] {src} -> {dst}")
            if content_faker_state is not None:
                content_faker_state.consume_new_identifiers()
            if name_faker_state is not None:
                name_faker_state.consume_new_identifiers()
            return (1, 0, 0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(redacted, encoding=enc)
        if content_faker_state is not None:
            content_faker_state.consume_new_identifiers()
        if name_faker_state is not None:
            name_faker_state.consume_new_identifiers()
        return (1, 0, 0)

    is_text_like = (src.suffix.lower() in TEXT_EXTS) or is_text_file(src, include_all)
    if not is_text_like:
        return _handle_nontext()
    if src.suffix.lower() == ".json":
        return _process_json_file()
    return _process_text_file()


def run_redaction(
    *,
    in_dir: Path,
    out_dir: Path,
    jobs: int,
    lang: str,
    entities: Optional[List[str]],
    score_threshold: float,
    operator: str,
    replace_with: str,
    mask_char: str,
    mask_chars_to_mask: int,
    mask_from_end: bool,
    allow_list: Optional[List[str]],
    allow_list_match: str,
    name_entities: Optional[List[str]],
    name_threshold: float,
    name_operator: str,
    name_replace_with: str,
    name_mask_char: str,
    name_mask_chars_to_mask: int,
    name_mask_from_end: bool,
    name_allow_list: Optional[List[str]],
    name_allow_list_match: str,
    chunk_size: int,
    chunk_break_window: int,
    spacy_max_length: Optional[int],
    include_all: bool,
    skip_nontext: bool,
    overwrite: bool,
    names_only: bool,
    content_only: bool,
    generic_json_strings: bool,
    dry_run: bool,
    no_progress: bool,
    verbose: bool,
    faker_locale: Optional[str] = None,
) -> Dict[str, int]:
    """
    Redact PII in both file contents and path names using Microsoft Presidio.
    Mirrors directory structure under out_dir with anonymized names.

    **Special behavior for parsed JSON outputs**:
    - If a file is *.json* and conforms to the parsed schema
      { "messages": [ { "role": "...", "content": "..." }, ... ], ... }
      then only messages[*].content strings are anonymized. Metadata is left as-is.
    """
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = _build_analyzer(lang, spacy_max_length)
    locale = (
        faker_locale.strip()
        if isinstance(faker_locale, str) and faker_locale.strip()
        else None
    )
    anonymizer = AnonymizerEngine()
    name_faker_state = FakerState(locale) if name_operator == "faker" else None

    # Build blocklist terms from transcripts/metadata.csv 'Contact' column(s)
    blocklist_terms: List[str] = _load_contact_identifiers(
        Path("transcripts/metadata.csv"), verbose
    )

    # Optional spaCy max_length bump (chunking should usually suffice)
    try:
        nlp = analyzer.nlp_engine.get_nlp(lang)
        if nlp is not None:
            default_target = max(1_200_000, chunk_size * 2)
            nlp.max_length = (
                int(spacy_max_length)
                if spacy_max_length
                else max(nlp.max_length, default_target)
            )
    except (AttributeError, TypeError, ValueError):
        # Non-fatal; chunking will still protect us
        pass

    content_op_params = op_params_for(
        operator, replace_with, mask_char, mask_chars_to_mask, mask_from_end
    )
    name_op_params = op_params_for(
        name_operator,
        name_replace_with,
        name_mask_char,
        name_mask_chars_to_mask,
        name_mask_from_end,
    )
    # Use name-based params to de-identify directory and file names as well

    # Map original dirs to anonymized destinations so children resolve properly
    dir_map: Dict[Path, Path] = {in_dir: out_dir}

    # First pass: create anonymized directory tree (parents before children)
    src_dirs = sorted(
        [p for p in in_dir.rglob("*") if p.is_dir()],
        key=lambda p: len(p.relative_to(in_dir).parts),
    )
    for d in src_dirs:
        parent_dst = dir_map.get(d.parent, out_dir / d.parent.relative_to(in_dir))
        # Anonymize the directory name using Presidio unless content-only
        if content_only:
            anon_name = d.name
        else:
            # Preserve exact dirs like hl_123, irb_456, or 105; anonymize others
            if re.fullmatch(r"(hl_\d+|irb_\d+|[12]\d+)", d.name):
                anon_name = d.name
            else:
                anon_name = _anonymize_tokenwise(
                    text=d.name,
                    analyzer=analyzer,
                    anonymizer=anonymizer,
                    language=lang,
                    name_entities=name_entities,
                    name_threshold=name_threshold,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_allow_list=name_allow_list,
                    name_allow_list_match=name_allow_list_match,
                    name_faker_state=name_faker_state,
                )
        dir_fallback = d.name if name_faker_state is not None else "REDACTED"
        anon_name = safe_fs_component(
            anon_name,
            fallback=dir_fallback,
            faker_state=name_faker_state,
        )
        if overwrite:
            anon_name = unique_name_in(parent_dst, anon_name)
        dst_dir = parent_dst / anon_name
        dir_map[d] = dst_dir

        if verbose:
            print(f"[DIR] {d} -> {dst_dir}")
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)

    # Second pass: prepare file tasks shared by serial and parallel paths
    files = [p for p in in_dir.rglob("*") if p.is_file()]
    total_files = len(files)
    processed_text = 0
    copied_bin = 0
    failed = 0
    tasks: List[Dict[str, object]] = []
    skipped_exists = 0

    for src in files:
        try:
            parent_dst = dir_map.get(
                src.parent, out_dir / src.parent.relative_to(in_dir)
            )
            if parent_dst is None:
                parent_dst = out_dir / src.parent.relative_to(in_dir)
            dst = _destination_for_file(
                src=src,
                parent_dst=parent_dst,
                content_only=content_only,
                analyzer=analyzer,
                anonymizer=anonymizer,
                language=lang,
                name_entities=name_entities,
                name_threshold=name_threshold,
                name_operator=name_operator,
                name_op_params=name_op_params,
                name_allow_list=name_allow_list,
                name_allow_list_match=name_allow_list_match,
                name_faker_state=name_faker_state,
                overwrite=overwrite,
            )
            if dst.exists() and not overwrite:
                skipped_exists += 1
                if verbose or dry_run:
                    print(f"[SKIP-EXISTS] {src} -> {dst}")
                continue
            task = {
                "src": str(src),
                "dst": str(dst),
                "lang": lang,
                "spacy_max_length": spacy_max_length,
                "chunk_size": chunk_size,
                "chunk_break_window": chunk_break_window,
                "include_all": include_all,
                "skip_nontext": skip_nontext,
                "names_only": names_only,
                "generic_json_strings": generic_json_strings,
                "entities": entities,
                "score_threshold": score_threshold,
                "operator": operator,
                "content_op_params": content_op_params,
                "allow_list": allow_list,
                "allow_list_match": allow_list_match,
                "name_entities": name_entities,
                "name_threshold": name_threshold,
                "name_operator": name_operator,
                "name_op_params": name_op_params,
                "name_allow_list": name_allow_list,
                "name_allow_list_match": name_allow_list_match,
                "content_use_faker": operator == "faker",
                "name_use_faker": name_operator == "faker",
                "faker_locale": locale,
                "dry_run": dry_run,
                "verbose": verbose,
                "no_progress": True if jobs and jobs > 1 else no_progress,
                "blocklist_terms": blocklist_terms,
            }
            tasks.append(task)
        except (OSError, RuntimeError, ValueError) as exc:
            failed += 1
            print(f"[WARN] Failed to prepare {src}: {exc}", file=sys.stderr)

    if jobs and jobs > 1:
        if verbose:
            print(
                f"[INFO] Files scheduled: {len(tasks)}, skipped existing: {skipped_exists}"
            )
        if not tasks:
            return {
                "text": processed_text,
                "bin": copied_bin,
                "failed": failed,
                "total": total_files,
            }
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            fut_to_task = {pool.submit(_process_file_task, t): t for t in tasks}
            fut_list = list(fut_to_task.keys())
            _ = (
                fut_list if no_progress else tqdm(fut_list, desc="Files", unit="file")
            )  # keeps progress identical to previous behavior
            try:
                for fut in as_completed(fut_to_task):
                    t = fut_to_task[fut]
                    try:
                        tp, bc, fl = fut.result()
                        processed_text += tp
                        copied_bin += bc
                        failed += fl
                        if verbose and fl == 0 and (tp + bc) > 0:
                            print(f"[OK] {t['src']} -> {t['dst']}")
                    except (
                        OSError,
                        UnicodeError,
                        json.JSONDecodeError,
                        RuntimeError,
                        ValueError,
                    ) as ex:
                        failed += 1
                        if verbose:
                            print(f"[FAIL] {t['src']}: {ex}", file=sys.stderr)
                        else:
                            print(f"[WARN] worker failed: {ex}", file=sys.stderr)
            except KeyboardInterrupt:
                print(
                    "[INFO] Anonymization interrupted by user. Cancelling pending tasks...",
                    file=sys.stderr,
                )
                for fut in fut_to_task:
                    if not fut.done():
                        fut.cancel()
                for fut in fut_to_task:
                    if fut.done() and not fut.cancelled():
                        try:
                            tp, bc, fl = fut.result()
                            processed_text += tp
                            copied_bin += bc
                            failed += fl
                        except (
                            OSError,
                            UnicodeError,
                            json.JSONDecodeError,
                            RuntimeError,
                            ValueError,
                        ):
                            failed += 1
                return {
                    "text": processed_text,
                    "bin": copied_bin,
                    "failed": failed,
                    "total": total_files,
                }
        return {
            "text": processed_text,
            "bin": copied_bin,
            "failed": failed,
            "total": total_files,
        }

    iterator = tasks if no_progress else tqdm(tasks, desc="Files", unit="file")
    try:
        for idx, task in enumerate(iterator, start=1):
            src_path = Path(str(task["src"]))
            dst_path = Path(str(task["dst"]))
            if verbose:
                print(f"[FILE {idx}/{total_files}] {src_path} -> {dst_path}")
            try:
                params = _build_redaction_params_from_task(task)
                # Keep no_progress consistent with the outer loop behavior
                params["no_progress"] = no_progress
                tp, bc, fl = _redact_single_file(**params)
                processed_text += tp
                copied_bin += bc
                failed += fl
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                RuntimeError,
                ValueError,
            ) as exc:
                failed += 1
                if verbose:
                    print(f"[FAIL] {src_path}: {exc}", file=sys.stderr)
                else:
                    print(f"[WARN] failed: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print(
            "[INFO] Anonymization interrupted by user. Returning partial results...",
            file=sys.stderr,
        )
        return {
            "text": processed_text,
            "bin": copied_bin,
            "failed": failed,
            "total": total_files,
        }

    if verbose:
        print(f"[INFO] Skipped existing (serial): {skipped_exists}")

    return {
        "text": processed_text,
        "bin": copied_bin,
        "failed": failed,
        "total": total_files,
    }


def _find_installed_spacy_model_dir() -> Optional[Path]:
    """
    In a dev (non-frozen) environment, find the true model directory for en_core_web_lg.
    Some wheels install the data under a nested subdir, not at the package root.
    """
    base = Path(en_core_web_lg.__file__).resolve().parent
    if (base / "config.cfg").exists():
        return base
    # Search a bit deeper for config.cfg
    try:
        for q in base.rglob("config.cfg"):
            return q.parent
    except OSError:
        pass
    # Fallback: meta.json
    try:
        for q in base.rglob("meta.json"):
            return q.parent
    except OSError:
        pass
    return None


def _build_analyzer(lang: str, spacy_max_length: Optional[int]) -> AnalyzerEngine:
    """
    Build AnalyzerEngine using an installed spaCy model.

    Prefers loading by *package name* ('en_core_web_lg'), falling back to an
    installed model directory when needed. This avoids runtime downloads and
    the E053 path/config issue.
    """
    cache_key = (lang, int(spacy_max_length) if spacy_max_length else None, False)

    # Per-process cache: avoid rebuilding the heavy spaCy/Presidio analyzer
    # for every file when running with multiple workers.
    cached = _ANALYZER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def _bump_max_len(ae: AnalyzerEngine) -> None:
        if not spacy_max_length:
            return
        try:
            nlp_obj = ae.nlp_engine.nlp.get(lang)
            if nlp_obj is not None:
                nlp_obj.max_length = int(spacy_max_length)
        except (AttributeError, TypeError, ValueError):
            pass

    # Load by package *name* (most reliable for installed models)
    cfg = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": "en_core_web_lg"}],
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=cfg)
        nlp_engine = provider.create_engine()
        ae = AnalyzerEngine(nlp_engine=nlp_engine)
        _bump_max_len(ae)
        _ANALYZER_CACHE[cache_key] = ae
        return ae
    except (OSError, RuntimeError, ValueError, KeyError):
        # Fallback: locate the true installed model directory and load by path
        model_dir = _find_installed_spacy_model_dir()
        if model_dir:
            cfg = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": lang, "model_name": str(model_dir)}],
            }
            provider = NlpEngineProvider(nlp_configuration=cfg)
            nlp_engine = provider.create_engine()
            ae = AnalyzerEngine(nlp_engine=nlp_engine)
            _bump_max_len(ae)
            _ANALYZER_CACHE[cache_key] = ae
            return ae
        # Last fallback: let Presidio try defaults (may still work in some dev envs)
        ae = AnalyzerEngine()
        _bump_max_len(ae)
        _ANALYZER_CACHE[cache_key] = ae
        return ae


# (deduped) use util_anonymize_text from redaction_utils


def _is_structural_key(key_lower: str) -> bool:
    """Return True for common non-file structural key names."""
    if key_lower in {
        "id",
        "role",
        "content_type",
        "status",
        "recipient",
        "channel",
    }:
        return True
    if key_lower.endswith(("_id", "id", "_ids", "ids", "_slug", "slug")):
        return True
    return False


def _looks_like_path_or_url(value: str) -> bool:
    """Heuristically detect strings that look like paths or URLs."""
    return (
        "://" in value
        or value.startswith("sandbox:/")
        or "/" in value
        or "\\" in value
        or PATH_SUFFIX_RE.search(value) is not None
    )


def _safe_token(value: str) -> str:
    """Return a filesystem-safe token derived from ``value``."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return token or "REDACTED"


def _faker_token_for_fileish(
    value: str,
    *,
    name_faker_state: Optional[FakerState],
    content_faker_state: Optional[FakerState],
) -> Optional[str]:
    """Return a fake token for file-like strings using configured Faker states."""
    for state in (name_faker_state, content_faker_state):
        if state is None:
            continue
        fake = state.replacement("FILE", value)
        if isinstance(fake, str) and fake.strip():
            return _safe_token(fake)
    return None


def _fallback_fileish(value: str, token: str) -> str:
    """Construct a plausible replacement for a path or URL using ``token``."""
    token_safe = _safe_token(token)
    if "://" in value:
        prefix, rest = value.split("://", 1)
        ext = PATH_SUFFIX_RE.search(rest)
        ext_text = ext.group(0) if ext else ""
        return f"{prefix}://{token_safe}{ext_text}"
    if "/" in value:
        prefix, _sep, tail = value.rpartition("/")
        ext = PATH_SUFFIX_RE.search(tail)
        ext_text = ext.group(0) if ext else ""
        base = f"{prefix}/" if prefix else "/"
        return f"{base}{token_safe}{ext_text}"
    if "\\" in value:
        prefix, _sep, tail = value.rpartition("\\")
        ext = PATH_SUFFIX_RE.search(tail)
        ext_text = ext.group(0) if ext else ""
        base = f"{prefix}\\" if prefix else "\\"
        return f"{base}{token_safe}{ext_text}"
    ext = PATH_SUFFIX_RE.search(value)
    if ext:
        return f"{token_safe}{ext.group(0)}"
    return token_safe


def _sanitize_fileish_string(
    value: str,
    *,
    text_anonymizer: Callable[[str], str],
    operator: str,
    content_op_params: Dict,
    name_operator: str,
    name_op_params: Dict,
    name_faker_state: Optional[FakerState],
    content_faker_state: Optional[FakerState],
) -> str:
    """Anonymize a file-like string while preserving extension and shape."""
    new_value = text_anonymizer(value)
    if new_value != value:
        return new_value
    token = _faker_token_for_fileish(
        value,
        name_faker_state=name_faker_state,
        content_faker_state=content_faker_state,
    )
    if token is None:
        if operator == "replace":
            token = str(content_op_params.get("new_value", "REDACTED"))
        elif name_operator == "replace":
            token = str(name_op_params.get("new_value", "REDACTED"))
        else:
            token = "REDACTED"
    return _fallback_fileish(value, token)


def _maybe_anonymize_fileish_str(
    value: str,
    key_lower: Optional[str],
    *,
    file_key_tokens: Tuple[str, ...],
    text_anonymizer: Callable[[str], str],
    operator: str,
    content_op_params: Dict,
    name_operator: str,
    name_op_params: Dict,
    name_faker_state: Optional[FakerState],
    content_faker_state: Optional[FakerState],
) -> str:
    """Return anonymized value when key/value suggest file or URL semantics."""
    if key_lower and _is_structural_key(key_lower):
        return value
    key_trigger = key_lower and any(token in key_lower for token in file_key_tokens)
    if key_trigger or _looks_like_path_or_url(value):
        return _sanitize_fileish_string(
            value,
            text_anonymizer=text_anonymizer,
            operator=operator,
            content_op_params=content_op_params,
            name_operator=name_operator,
            name_op_params=name_op_params,
            name_faker_state=name_faker_state,
            content_faker_state=content_faker_state,
        )
    return value


def _process_fileish_list(
    seq: List,
    key_lower: Optional[str],
    *,
    file_key_tokens: Tuple[str, ...],
    text_anonymizer: Callable[[str], str],
    operator: str,
    content_op_params: Dict,
    name_operator: str,
    name_op_params: Dict,
    name_faker_state: Optional[FakerState],
    content_faker_state: Optional[FakerState],
) -> bool:
    """Recursively anonymize file-like strings inside a list."""
    changed = False
    hint_from_key = bool(
        key_lower
        and not _is_structural_key(key_lower)
        and any(token in key_lower for token in file_key_tokens)
    )
    new_items: List = []
    for item in seq:
        new_item = item
        if isinstance(item, str):
            if hint_from_key or _looks_like_path_or_url(item):
                new_item = _sanitize_fileish_string(
                    item,
                    text_anonymizer=text_anonymizer,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                )
        elif isinstance(item, dict):
            if _anonymize_fileish_fields(
                item,
                file_key_tokens=file_key_tokens,
                text_anonymizer=text_anonymizer,
                operator=operator,
                content_op_params=content_op_params,
                name_operator=name_operator,
                name_op_params=name_op_params,
                name_faker_state=name_faker_state,
                content_faker_state=content_faker_state,
            ):
                changed = True
        elif isinstance(item, list):
            if _process_fileish_list(
                item,
                key_lower,
                file_key_tokens=file_key_tokens,
                text_anonymizer=text_anonymizer,
                operator=operator,
                content_op_params=content_op_params,
                name_operator=name_operator,
                name_op_params=name_op_params,
                name_faker_state=name_faker_state,
                content_faker_state=content_faker_state,
            ):
                changed = True
        if new_item != item:
            changed = True
        new_items.append(new_item)
    if changed:
        seq[:] = new_items
    return changed


def _anonymize_fileish_fields(
    node,
    *,
    file_key_tokens: Tuple[str, ...],
    text_anonymizer: Callable[[str], str],
    operator: str,
    content_op_params: Dict,
    name_operator: str,
    name_op_params: Dict,
    name_faker_state: Optional[FakerState],
    content_faker_state: Optional[FakerState],
) -> bool:
    """Recursively anonymize file/path/URL-like fields in a JSON-like object."""
    if isinstance(node, dict):
        changed_local = False
        for key, value in list(node.items()):
            key_lower = key.lower()
            if isinstance(value, str):
                new_value = _maybe_anonymize_fileish_str(
                    value,
                    key_lower,
                    file_key_tokens=file_key_tokens,
                    text_anonymizer=text_anonymizer,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                )
                if new_value != value:
                    node[key] = new_value
                    changed_local = True
            elif isinstance(value, list):
                if _process_fileish_list(
                    value,
                    key_lower,
                    file_key_tokens=file_key_tokens,
                    text_anonymizer=text_anonymizer,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                ):
                    changed_local = True
            elif isinstance(value, dict):
                if _anonymize_fileish_fields(
                    value,
                    file_key_tokens=file_key_tokens,
                    text_anonymizer=text_anonymizer,
                    operator=operator,
                    content_op_params=content_op_params,
                    name_operator=name_operator,
                    name_op_params=name_op_params,
                    name_faker_state=name_faker_state,
                    content_faker_state=content_faker_state,
                ):
                    changed_local = True
        return changed_local
    if isinstance(node, list):
        return _process_fileish_list(
            node,
            None,
            file_key_tokens=file_key_tokens,
            text_anonymizer=text_anonymizer,
            operator=operator,
            content_op_params=content_op_params,
            name_operator=name_operator,
            name_op_params=name_op_params,
            name_faker_state=name_faker_state,
            content_faker_state=content_faker_state,
        )
    return False


def _anonymize_json_strings(
    obj,
    *,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    language: str,
    entities: Optional[List[str]],
    score_threshold: float,
    operator: str,
    op_params: Dict,
    allow_list: Optional[List[str]],
    allow_list_match: str,
    chunk_size: int,
    chunk_break_window: int,
    faker_state: Optional[FakerState],
) -> int:
    """Recursively anonymize only string values inside a JSON-compatible object.

    Returns an approximate count of updated string fields.
    """
    changed = 0

    def _walk(node):
        nonlocal changed
        if isinstance(node, dict):
            for k, v in list(node.items()):
                node[k] = _walk(v)
            return node
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            text = node
            new_text = util_anonymize_chunked_text(
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
                chunk_size=chunk_size,
                chunk_break_window=chunk_break_window,
            )
            if new_text != text:
                changed += 1
            return new_text
        return node

    _walk(obj)
    return changed
