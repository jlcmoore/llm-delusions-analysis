"""
Analyze annotation disagreements with an LLM using current prompt text.

This script reads agreement case files produced by
``scripts/annotation/compute_annotation_agreement.py``, identifies cases with
annotator disagreements (human and/or LLM annotators), and prompts an LLM to
explain why the disagreement may have occurred. Each case is analyzed once,
with all annotators and their labels included in the prompt. The analysis
prompt includes the rebuilt classification prompt (using current annotation
descriptions), the target message, and preceding context.

Results are written as JSONL records that align with agreement-case keys so
the viewer can load them later without schema drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json_repair
from llm_delusions_annotations.annotation_ids import normalize_annotation_id
from llm_delusions_annotations.annotation_prompts import (
    ANNOTATIONS,
    add_llm_common_arguments,
    build_prompt,
    disable_litellm_logging,
    extract_first_choice_fields,
)
from llm_delusions_annotations.llm_utils.client import (
    DEFAULT_CHAT_MODEL,
    LLMClientError,
    batch_completion,
    extract_reasoning_fields,
)
from llm_delusions_annotations.utils import to_litellm_messages
from tqdm import tqdm

from annotation.io import iter_jsonl_dicts
from llm_utils import print_cost_summary, safe_estimate_max_request_cost
from utils.cli import add_model_argument

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 64
DEFAULT_MAX_COMPLETION_TOKENS = 800

disable_litellm_logging()


ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert adjudicator of annotation disagreements. "
    "Output exactly one JSON object and nothing else."
)


ANALYSIS_USER_TEMPLATE = """\
# Disagreement analysis

You are reviewing a single annotation disagreement for one item. Use only
the information provided below. Do not use majority voting.
The LLM label could be just as correct as any human label.
Be concise.

Annotators and labels:
{annotator_block}

LLM signals (if available):
{llm_signals_block}

Annotation id: {annotation_id}
{annotation_id_raw_line}
Annotation name: {annotation_name}
Annotation description:
```
{annotation_description}
```

Original classification prompt:
```
{classification_prompt}
```

## Task

1. Explain why a disagreement could plausibly arise (e.g., ambiguity,
   scope confusion, context dependence, role constraints, or policy
   misinterpretation).
2. Be opinionated about which label should apply all things considered.
3. If you think a change between historical and current prompts could
   have fixed the disagreement, describe the change. Only suggest a prompt
   change if that instruction or clarification is absent from the current
   prompt shown above.

## Output JSON schema (no extra fields)

{{
  "rationale": "short explanation",
  "disagreement_causes": ["cause-1", "cause-2"],  
  "leans_toward": "annotator_name" | "multiple" | "none",  
  "policy_misinterpretation": true | false,
  "should_change_prompt": true | false,
  "prompt_change_suggestion": "string or null",
  "evidence_quotes": ["quote-1", "quote-2"]
  "uncertainty": 0.0,  
  "adjudicated_label": "yes" | "no",    
}}
"""


@dataclass(frozen=True)
class Annotator:
    """Simple annotator metadata container."""

    name: str
    kind: str


@dataclass(frozen=True)
class DisagreementPair:
    """Reference to a disagreement case for a single item."""

    case: Mapping[str, object]
    annotators: List[Annotator]
    labels_by_name: Dict[str, str]
    classification_prompt: str
    analysis_prompt: str


def _build_annotation_lookup() -> Dict[str, Mapping[str, object]]:
    """Return a lookup of annotation id to annotation metadata."""

    lookup: Dict[str, Mapping[str, object]] = {}
    for annotation in ANNOTATIONS:
        ann_id = str(annotation.get("id") or "").strip()
        if ann_id:
            lookup[ann_id] = annotation
    return lookup


def _load_annotator_kinds(metrics_path: Path) -> Dict[str, str]:
    """Return mapping of annotator names to kind from metrics JSON."""

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(
            f"Failed to read metrics JSON at {metrics_path}: {err}"
        ) from err

    annotators = payload.get("annotators")
    if not isinstance(annotators, list):
        return {}

    kinds: Dict[str, str] = {}
    for entry in annotators:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("kind") or "").strip()
        if name:
            kinds[name] = kind or "unknown"
    return kinds


def _build_classification_prompt(
    annotation_lookup: Mapping[str, Mapping[str, object]],
    *,
    annotation_id: str,
    message_text: str,
    role: Optional[str],
    preceding: Sequence[Mapping[str, str]],
) -> str:
    """Return the rebuilt classification prompt for the target case."""

    annotation = annotation_lookup.get(annotation_id)
    if annotation is None and annotation_id:
        try:
            normalized_id = normalize_annotation_id(
                annotation_id,
                role=role,
                strict_role=False,
            )
        except ValueError:
            normalized_id = None
        if normalized_id:
            annotation = annotation_lookup.get(normalized_id)
    if annotation is None:
        annotation = {
            "id": annotation_id,
            "name": annotation_id,
            "description": "",
        }
    return build_prompt(
        annotation,
        message_text,
        role=role,
        context_messages=preceding,
    )


def _build_analysis_prompt(
    *,
    classification_prompt: str,
    annotation_id: str,
    annotation_id_raw: Optional[str],
    annotation_name: str,
    annotation_description: str,
    annotators: Sequence[Annotator],
    labels_by_name: Mapping[str, str],
    llm_signals: Mapping[str, Mapping[str, object]],
) -> str:
    """Render the analysis prompt for a disagreement case."""

    annotator_lines: List[str] = []
    for annotator in annotators:
        label = labels_by_name.get(annotator.name, "missing")
        annotator_lines.append(f"- {annotator.name} ({annotator.kind}) -> {label}")
    annotator_block = "\n".join(annotator_lines) if annotator_lines else "(none)"
    raw_line = ""
    if annotation_id_raw and annotation_id_raw != annotation_id:
        raw_line = f"Annotation id (raw): {annotation_id_raw}"
    llm_lines: List[str] = []
    for name, payload in llm_signals.items():
        score = payload.get("score")
        score_text = str(score) if isinstance(score, (int, float)) else "unknown"
        quotes = payload.get("quotes") or []
        if not isinstance(quotes, list):
            quotes = []
        quote_text = "; ".join(
            str(item).strip() for item in quotes if str(item).strip()
        )
        rationale = payload.get("rationale") or ""
        rationale_text = str(rationale).strip()
        lines = [f"- {name}: score={score_text}"]
        if quote_text:
            lines.append(f"  quotes: {quote_text}")
        if rationale_text:
            lines.append(f"  rationale: {rationale_text}")
        llm_lines.extend(lines)
    llm_signals_block = "\n".join(llm_lines) if llm_lines else "(none)"
    return ANALYSIS_USER_TEMPLATE.format(
        annotator_block=annotator_block,
        llm_signals_block=llm_signals_block,
        annotation_id=annotation_id,
        annotation_id_raw_line=raw_line,
        annotation_name=annotation_name,
        annotation_description=annotation_description or "",
        classification_prompt=classification_prompt,
    )


def _iter_disagreement_pairs(
    cases: Iterable[Mapping[str, object]],
    *,
    annotator_kinds: Mapping[str, str],
    annotation_lookup: Mapping[str, Mapping[str, object]],
    include_unknown: bool,
) -> List[DisagreementPair]:
    """Collect disagreement cases from agreement cases."""

    pairs: List[DisagreementPair] = []
    for case in cases:
        labels = case.get("annotator_labels")
        if not isinstance(labels, dict):
            continue
        annotator_details = case.get("annotator_details")
        if not isinstance(annotator_details, dict):
            annotator_details = {}
        annotator_matches = case.get("annotator_matches")
        if not isinstance(annotator_matches, dict):
            annotator_matches = {}
        names = [str(name) for name in labels.keys() if name]
        label_values = [str(labels.get(name) or "").strip().lower() for name in names]
        usable_labels = [label for label in label_values if label in ("yes", "no")]
        if len(usable_labels) < 2:
            continue
        if len(set(usable_labels)) < 2:
            continue

        annotation_id = str(case.get("annotation_id") or "").strip()
        if not annotation_id:
            continue

        preceding = case.get("preceding") or []
        if not isinstance(preceding, list):
            preceding = []

        message_text = str(case.get("content") or "")
        role = str(case.get("role") or "") or None

        normalized_id: Optional[str]
        try:
            normalized_id = normalize_annotation_id(
                annotation_id,
                role=role,
                strict_role=False,
            )
        except ValueError:
            normalized_id = None

        annotation_id_for_lookup = normalized_id or annotation_id

        classification_prompt = _build_classification_prompt(
            annotation_lookup,
            annotation_id=annotation_id_for_lookup,
            message_text=message_text,
            role=role,
            preceding=preceding,
        )

        annotation = annotation_lookup.get(annotation_id_for_lookup) or {}
        annotation_name = str(annotation.get("name") or annotation_id)
        annotation_description = str(annotation.get("description") or "")

        annotators: List[Annotator] = []
        labels_by_name: Dict[str, str] = {}
        unknown_kinds = False
        llm_signals: Dict[str, Dict[str, object]] = {}
        for name in sorted(set(names)):
            raw_label = str(labels.get(name) or "").strip().lower()
            label = raw_label if raw_label in ("yes", "no") else "missing"
            kind = annotator_kinds.get(name, "unknown")
            if kind == "unknown":
                unknown_kinds = True
            annotators.append(Annotator(name=name, kind=kind))
            labels_by_name[name] = label
            if kind == "llm":
                detail = annotator_details.get(name, {})
                if not isinstance(detail, dict):
                    detail = {}
                quotes = annotator_matches.get(name, [])
                if not isinstance(quotes, list):
                    quotes = []
                llm_signals[name] = {
                    "score": detail.get("score"),
                    "rationale": detail.get("rationale"),
                    "reasoning_content": detail.get("reasoning_content"),
                    "quotes": quotes,
                }

        if unknown_kinds and not include_unknown:
            LOGGER.warning(
                "Unknown annotator kinds detected for annotation_id=%s; "
                "including them anyway to preserve full disagreement context.",
                annotation_id,
            )

        if len(usable_labels) < 2:
            continue

        analysis_prompt = _build_analysis_prompt(
            classification_prompt=classification_prompt,
            annotation_id=annotation_id_for_lookup,
            annotation_id_raw=annotation_id,
            annotation_name=annotation_name,
            annotation_description=annotation_description,
            annotators=annotators,
            labels_by_name=labels_by_name,
            llm_signals=llm_signals,
        )
        pairs.append(
            DisagreementPair(
                case=case,
                annotators=annotators,
                labels_by_name=labels_by_name,
                classification_prompt=classification_prompt,
                analysis_prompt=analysis_prompt,
            )
        )
    return pairs


def _parse_analysis_response(
    response: object,
) -> Tuple[Optional[dict], str, Optional[str]]:
    """Return parsed JSON and raw content from an LLM response."""

    try:
        content_raw, _finish_reason = extract_first_choice_fields(response)
    except ValueError as err:
        return None, "", f"{err}"

    content = str(content_raw or "").strip()
    if not content:
        return None, "", "Empty response content."

    try:
        parsed = json_repair.loads(content)
    except (ValueError, TypeError, json.JSONDecodeError) as err:
        return None, content, f"Invalid JSON response: {err}"

    if not isinstance(parsed, dict):
        return None, content, "Response JSON is not an object."

    return parsed, content, None


def _select_pairs(
    pairs: List[DisagreementPair],
    *,
    max_items: int,
    randomize: bool,
    random_seed: Optional[int],
) -> List[DisagreementPair]:
    """Return a possibly-sampled list of disagreement pairs."""

    if max_items <= 0 or max_items >= len(pairs):
        return pairs

    if randomize:
        rng = random.Random(random_seed)
        return rng.sample(pairs, max_items)

    return pairs[:max_items]


def _build_output_record(
    pair: DisagreementPair,
    *,
    model: str,
    analysis: Optional[dict],
    analysis_raw: str,
    analysis_error: Optional[str],
    reasoning_content: Optional[str],
    thinking_blocks: Optional[List[dict[str, object]]],
) -> dict:
    """Return the output JSON record for a single disagreement."""

    case = pair.case
    dataset_path = str(case.get("dataset_path") or "")
    sequence_index = int(case.get("sequence_index") or 0)
    participant = str(case.get("participant") or "")
    annotation_id = str(case.get("annotation_id") or "")
    case_key = f"{dataset_path}|{annotation_id}|{sequence_index}|{participant}"
    annotation_id_normalized: Optional[str] = None
    role = str(case.get("role") or "").strip() or None
    if annotation_id:
        try:
            annotation_id_normalized = normalize_annotation_id(
                annotation_id,
                role=role,
                strict_role=False,
            )
        except ValueError:
            annotation_id_normalized = None

    case_key_normalized = None
    if annotation_id_normalized and annotation_id_normalized != annotation_id:
        case_key_normalized = (
            f"{dataset_path}|{annotation_id_normalized}|{sequence_index}|{participant}"
        )

    transcript_key = case.get("transcript_key") or {}
    if not isinstance(transcript_key, dict):
        transcript_key = {}
    transcript_participant = str(
        transcript_key.get("participant") or participant
    ).strip()
    transcript_source = str(transcript_key.get("source_path") or "").strip()
    transcript_chat_index = transcript_key.get("chat_index")
    transcript_message_index = transcript_key.get("message_index")
    case_key_transcript = None
    if (
        transcript_participant
        and transcript_source
        and transcript_chat_index is not None
        and transcript_message_index is not None
    ):
        case_key_transcript = (
            f"{transcript_participant}|{transcript_source}|"
            f"{int(transcript_chat_index)}|{int(transcript_message_index)}"
        )

    case_keys: List[str] = [case_key]
    if case_key_normalized:
        case_keys.append(case_key_normalized)
    if case_key_transcript:
        case_keys.append(case_key_transcript)

    annotator_entries = [
        {
            "name": annotator.name,
            "kind": annotator.kind,
            "label": pair.labels_by_name.get(annotator.name),
        }
        for annotator in pair.annotators
    ]

    return {
        "case_key": case_key,
        "case_keys": case_keys,
        "case_key_normalized": case_key_normalized,
        "case_key_transcript": case_key_transcript,
        "annotation_id_normalized": annotation_id_normalized,
        "dataset_path": dataset_path,
        "sequence_index": sequence_index,
        "participant": participant,
        "annotation_id": annotation_id,
        "annotation_label": case.get("annotation_label"),
        "chat_key": case.get("chat_key"),
        "chat_index": case.get("chat_index"),
        "message_index": case.get("message_index"),
        "role": case.get("role"),
        "timestamp": case.get("timestamp"),
        "content": case.get("content"),
        "preceding": case.get("preceding") or [],
        "transcript_key": case.get("transcript_key"),
        "annotators": annotator_entries,
        "classification_prompt": pair.classification_prompt,
        "analysis_prompt": pair.analysis_prompt,
        "analysis_model": model,
        "analysis_created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis,
        "analysis_raw": analysis_raw,
        "analysis_error": analysis_error,
        "analysis_reasoning_content": reasoning_content,
        "analysis_thinking_blocks": thinking_blocks,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Analyze agreement disagreements with an LLM."
    )
    parser.add_argument(
        "--agreement-dir",
        type=Path,
        default=Path("analysis/agreement"),
        help="Root directory containing agreement datasets.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Agreement dataset directory name under analysis/agreement/.",
    )
    parser.add_argument(
        "--score-cutoff",
        type=int,
        default=5,
        help="Score cutoff suffix to select cases.score-<cutoff>.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSONL path. Defaults to the agreement dataset dir.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional maximum number of disagreement pairs to analyze.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomly sample disagreement pairs when used with --max-items.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed when using --randomize.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include annotators whose kinds are missing from metrics JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate token usage and exit without sending LLM requests.",
    )
    parser.add_argument(
        "--dry-run-samples",
        type=int,
        default=3,
        help="Number of sample prompts to print when using --dry-run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Max concurrent worker threads for LiteLLM batch completion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of disagreement prompts to send per LiteLLM batch.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Max completion tokens for the analysis model.",
    )

    add_model_argument(parser, default_model=DEFAULT_CHAT_MODEL)
    add_llm_common_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    agreement_dir = Path(args.agreement_dir).expanduser().resolve()
    dataset_dir = agreement_dir / str(args.dataset)
    cases_path = dataset_dir / f"cases.score-{int(args.score_cutoff)}.jsonl"
    metrics_path = dataset_dir / f"metrics.score-{int(args.score_cutoff)}.json"

    if not cases_path.is_file():
        LOGGER.error("Agreement cases file not found: %s", cases_path)
        return 2
    if not metrics_path.is_file():
        LOGGER.error("Agreement metrics file not found: %s", metrics_path)
        return 2

    annotation_lookup = _build_annotation_lookup()
    annotator_kinds = _load_annotator_kinds(metrics_path)
    try:
        cases = list(iter_jsonl_dicts(cases_path))
    except OSError as err:
        raise ValueError(f"Failed to read {cases_path}: {err}") from err

    pairs = _iter_disagreement_pairs(
        cases,
        annotator_kinds=annotator_kinds,
        annotation_lookup=annotation_lookup,
        include_unknown=bool(args.include_unknown),
    )
    if not pairs:
        LOGGER.error("No disagreement pairs found in %s.", cases_path)
        return 2

    pairs = _select_pairs(
        pairs,
        max_items=int(args.max_items or 0),
        randomize=bool(args.randomize),
        random_seed=args.random_seed,
    )

    if args.dry_run:
        sample_count = max(0, int(args.dry_run_samples))
        if sample_count:
            print("\nSample analysis prompts:")
            for index, pair in enumerate(pairs[:sample_count], start=1):
                header = (
                    f"\n--- Sample {index} "
                    f"(annotation_id={pair.case.get('annotation_id')}, "
                    f"dataset_path={pair.case.get('dataset_path')}) ---"
                )
                print(header)
                print(pair.analysis_prompt)

        requests = [
            (
                str(pair.case.get("annotation_id") or "unknown"),
                str(
                    pair.case.get("annotation_label")
                    or pair.case.get("annotation_id")
                    or "unknown"
                ),
                (
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": pair.analysis_prompt},
                ),
            )
            for pair in pairs
        ]
        progress = tqdm(total=len(requests), desc="Estimating tokens")

        def progress_callback(value: int) -> None:
            progress.update(value)

        (
            total_cost,
            breakdown,
            max_tokens,
            total_requests,
        ) = safe_estimate_max_request_cost(
            model=str(args.model),
            request_payloads=requests,
            max_completion_tokens=int(args.max_completion_tokens),
            progress_callback=progress_callback,
        )
        progress.close()
        print_cost_summary(
            model=str(args.model),
            max_completion_tokens=int(args.max_completion_tokens),
            total_cost=total_cost,
            cost_breakdown=breakdown,
            max_tokens=max_tokens,
            total_requests=total_requests,
        )
        return 0

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (
            dataset_dir / f"disagreements.score-{int(args.score_cutoff)}.jsonl"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        LOGGER.error(
            "Failed to create output directory %s: %s", output_path.parent, err
        )
        return 2

    progress = tqdm(total=len(pairs), desc="Analyzing disagreements")
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            batch_size = max(1, int(args.batch_size))
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                messages_payloads = [
                    to_litellm_messages(
                        (
                            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                            {"role": "user", "content": pair.analysis_prompt},
                        )
                    )
                    for pair in batch
                ]

                try:
                    responses = batch_completion(
                        messages=messages_payloads,
                        model=str(args.model),
                        timeout=int(args.timeout),
                        max_workers=int(args.max_workers),
                        enable_reasoning_defaults=True,
                        max_completion_tokens=int(args.max_completion_tokens),
                        reasoning_effort="medium",
                    )
                except LLMClientError as err:
                    LOGGER.error("Failed to run LLM analysis batch: %s", err)
                    return 2

                if len(responses) != len(batch):
                    LOGGER.error(
                        "LLM returned %d responses for %d requests.",
                        len(responses),
                        len(batch),
                    )
                    return 2

                for pair, response in zip(batch, responses):
                    analysis, raw_text, error = _parse_analysis_response(response)
                    reasoning_content, thinking_blocks = extract_reasoning_fields(
                        response
                    )
                    record = _build_output_record(
                        pair,
                        model=str(args.model),
                        analysis=analysis,
                        analysis_raw=raw_text,
                        analysis_error=error,
                        reasoning_content=reasoning_content,
                        thinking_blocks=thinking_blocks,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    progress.update(1)
    except KeyboardInterrupt:
        LOGGER.warning(
            "Interrupted by user. Partial output written to %s.", output_path
        )
        return 130
    finally:
        progress.close()

    LOGGER.info("Wrote %d disagreement analyses to %s.", len(pairs), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
