"""HTTP server for the classification viewer with simple JSON APIs.

This augments a no-cache static file server with a few endpoints that the
viewer can call locally. It intentionally does not expose any endpoint that can
trigger new LLM classifications and is intended only for inspecting existing
outputs during local development (see ``make viewer``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import warnings
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Final, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq
from llm_delusions_annotations.annotation_prompts import (
    ANNOTATIONS_FILE,
    BASE_SCOPE_TEXT,
)
from llm_delusions_annotations.chat.chat_utils import (
    compute_previous_indices_skipping_roles,
)
from llm_delusions_annotations.chat.timestamps import normalize_timestamp_value
from llm_delusions_annotations.configs import LLM_SCORE_CUTOFF
from llm_delusions_annotations.cutoffs import CUTOFFS_FILE

from annotation.annotation_tables import (
    LOCATION_WITH_CONTEXT_COLUMNS,
    build_content_mapping_for_locations,
)
from annotation.io import iter_annotation_output_runs, iter_jsonl_dicts
from llm_delusions_data import get_path
from llm_delusions_data.loaders import (
    load_annotations_matches_parquet,
    load_annotations_preprocessed_parquet,
    load_transcripts_parquet,
)
from utils.param_strings import string_to_dict

# Local imports are inside handlers to keep import costs low for static traffic.

CACHE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheRequestHandler(SimpleHTTPRequestHandler):
    """Serve files and handle small JSON API calls for the viewer."""

    def end_headers(self) -> None:
        for header, value in CACHE_HEADERS.items():
            self.send_header(header, value)
        super().end_headers()

    # -------- Utilities --------
    def _serve_package_file(self, traversable_path, content_type="text/plain") -> None:
        try:
            content = traversable_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Not found: {e}".encode("utf-8"))

    def _send_json(self, payload: object, status: int = 200) -> None:
        """Write a JSON response with the given HTTP status code.

        Parameters
        ----------
        payload: object
            JSON-serializable value to write.
        status: int
            HTTP status code (defaults to 200).
        """

        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as err:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"Failed to encode JSON: {err}"}).encode("utf-8")
            )
            return

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        """Parse and return the JSON body of a request.

        Returns
        -------
        dict
            Parsed JSON object.
        """

        length_str = self.headers.get("Content-Length", "0")
        try:
            length = int(length_str)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON: {err}") from err
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _iter_label_objects(self, path: Path) -> Iterable[dict]:
        """Yield JSON label dicts from a newline-delimited JSONL file."""

        try:
            yield from iter_jsonl_dicts(path)
        except OSError as err:
            raise OSError(f"Failed to read labels: {err}") from err

    @staticmethod
    def _normalize_annotator_id(raw_id: str) -> str:
        """Return a filesystem-safe, lowercase annotator id."""

        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_id.strip().lower())
        return cleaned or "anon"

    # -------- Routing --------
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        """Serve GET for static assets and small JSON endpoints."""

        if self.path == "/api/annotations.csv":
            self._serve_package_file(ANNOTATIONS_FILE, "text/csv")
            return
        if self.path == "/api/cutoffs.csv":
            self._serve_package_file(CUTOFFS_FILE, "text/csv")
            return
        if self.path == "/api/classify-defaults":
            self._handle_get_classify_defaults()
            return
        if self.path == "/api/config":
            self._handle_get_config()
            return
        if self.path.startswith("/api/classify-metadata"):
            self._handle_get_classify_metadata()
            return
        if self.path.startswith("/api/classify-records"):
            self._handle_get_classify_records()
            return
        if self.path == "/api/classify-datasets":
            self._handle_get_classify_datasets()
            return
        if self.path == "/api/agreement-datasets":
            self._handle_get_agreement_datasets()
            return
        if self.path == "/api/llm-runs":
            self._handle_get_llm_runs()
            return
        if self.path.startswith("/api/manual-labels"):
            self._handle_get_manual_labels()
            return
        if self.path == "/api/manual-datasets":
            self._handle_get_manual_datasets()
            return
        if self.path == "/api/manual-instructions":
            self._handle_get_manual_instructions()
            return
        super().do_GET()
        return

    def do_post(self) -> None:
        """Serve POST endpoints for annotations and classification.

        Note: The HTTP server expects a method named ``do_POST``. To satisfy
        linting rules for snake_case while maintaining compatibility, an alias
        to this method named ``do_POST`` is installed at module import time.
        """

        if self.path == "/api/context-messages":
            self._handle_context_messages()
            return
        if self.path == "/api/save-manual-labels":
            self._handle_save_manual_labels()
            return
        self.send_error(404, "Unknown endpoint")

    # -------- Endpoint impls --------
    def _handle_get_config(self) -> None:
        """Return small configuration values used by viewers."""

        self._send_json({"llm_score_cutoff": LLM_SCORE_CUTOFF})

    def _handle_get_classify_defaults(self) -> None:
        """Return default parameter values for classify_chats."""

        try:
            cc = load_classify_chats()
            # Ask classify_chats for defaults by parsing with minimal required args
            # so argparse does not error on --input.
            args = cc.parse_args(["--input", ".", "--dry-run"])
            defaults = getattr(args, "_defaults", {})

            keys = [
                "model",
                "timeout",
                "follow_links",
                "prefilter_conversations",
                "max_messages",
                "randomize",
                "randomize_per_ppt",
                "randomize_conversations",
                "max_conversations",
                "reverse_conversations",
                "preceding_context",
            ]
            payload = {key: defaults.get(key) for key in keys}
            self._send_json({"defaults": payload})
        except (ImportError, OSError, ValueError, TypeError) as err:
            # Surface a concise message to the UI; details go to stderr.
            print(f"[server] classify-defaults error: {err}", file=sys.stderr)
            self._send_json({"error": str(err)}, status=500)

    def _handle_get_llm_runs(self) -> None:
        """Return a summary of available LLM classification runs."""

        root = Path("annotation_outputs")
        if not root.exists():
            self._send_json({"runs": []})
            return

        try:
            runs = list(iter_annotation_output_runs(root))
        except OSError as err:
            self._send_json({"error": str(err)}, status=500)
            return

        payload: List[dict[str, object]] = []
        for run in runs:
            payload.append(
                {
                    "path": str(run.rel_path).replace("\\", "/"),
                    "model": run.model,
                    "participants": list(run.participants),
                    "annotation_ids": list(run.annotation_ids),
                    "preceding_context": run.preceding_context,
                    "generated_at": run.generated_at,
                    "bucket": run.bucket,
                    "participant_dir": run.participant_dir,
                }
            )

            self._send_json({"runs": payload})

    def _handle_get_classify_datasets(self) -> None:
        """Return available classification datasets backed by Parquet tables."""

        root = get_path("annotations_preprocessed").parent
        if not root.exists() or not root.is_dir():
            self._send_json({"datasets": []})
            return

        payload: List[dict[str, object]] = []
        for path in sorted(root.glob("*__preprocessed.parquet")):
            stem = path.stem
            key, _, _ = stem.partition("__preprocessed")
            key = key or stem
            try:
                rel_pre = path.relative_to(Path("."))
            except ValueError:
                rel_pre = path
            preprocessed_rel = str(rel_pre).replace("\\", "/")

            matches_path = path.with_name(f"{key}__matches.parquet")
            if matches_path.exists():
                try:
                    rel_matches = matches_path.relative_to(Path("."))
                except ValueError:
                    rel_matches = matches_path
                matches_rel = str(rel_matches).replace("\\", "/")
            else:
                matches_rel = ""

            filename = f"{key}.jsonl"
            label_timestamp = self._format_manual_dataset_timestamp(
                None,
                filename,
            )
            label_params = self._format_manual_dataset_params(filename)
            label_parts: List[str] = []
            if label_timestamp:
                label_parts.append(label_timestamp)
            if label_params:
                label_parts.append(label_params)
            pretty_label = " - ".join(label_parts) if label_parts else key

            payload.append(
                {
                    "key": key,
                    "label": pretty_label,
                    "preprocessed_path": preprocessed_rel,
                    "matches_path": matches_rel,
                }
            )

        self._send_json({"datasets": payload})

    def _handle_get_classify_metadata(self) -> None:
        """Return participants, annotation ids, and cutoff info for a dataset."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        key_raw = (params.get("dataset") or params.get("key") or [""])[0].strip()
        if not key_raw:
            self._send_json({"error": "Missing dataset key"}, status=400)
            return
        if "/" in key_raw or "\\" in key_raw:
            self._send_json({"error": "Invalid dataset key"}, status=400)
            return

        root = get_path("annotations_preprocessed").parent
        preprocessed_path = root / f"{key_raw}__preprocessed.parquet"
        matches_path = root / f"{key_raw}__matches.parquet"
        if not preprocessed_path.exists() and not matches_path.exists():
            self._send_json({"error": "Dataset not found"}, status=404)
            return

        participants: List[str] = []
        annotation_ids: List[str] = []

        if preprocessed_path.exists():
            try:
                pf = pq.ParquetFile(preprocessed_path)
                col_names = list(pf.schema.names)
                score_cols = [
                    name
                    for name in col_names
                    if name.startswith("score__") and len(name) > len("score__")
                ]
                annotation_ids = sorted(
                    {name[len("score__") :] for name in score_cols},
                )
                if "participant" in col_names:
                    table = pf.read(columns=["participant"])
                    raw_values = table.column("participant").to_pylist()
                    participants = sorted(
                        {
                            str(value).strip()
                            for value in raw_values
                            if value not in (None, "")
                        }
                    )
            except (OSError, ValueError) as err:
                self._send_json(
                    {"error": f"Failed to inspect dataset metadata: {err}"},
                    status=500,
                )
                return

        has_matches = matches_path.exists()
        cutoffs_by_annotation: dict[str, List[int]] = {}
        if has_matches:
            try:
                pf_matches = pq.ParquetFile(matches_path)
                match_names = list(pf_matches.schema.names)
                if "score_cutoff" in match_names:
                    table = pf_matches.read(columns=["annotation_id", "score_cutoff"])
                    annot_values = table.column("annotation_id").to_pylist()
                    cutoff_values = table.column("score_cutoff").to_pylist()
                    tmp: dict[str, set[int]] = {}
                    for annot, cutoff in zip(annot_values, cutoff_values):
                        if annot is None or cutoff is None:
                            continue
                        annot_str = str(annot).strip()
                        if not annot_str:
                            continue
                        try:
                            cutoff_int = int(cutoff)
                        except (TypeError, ValueError):
                            continue
                        if annot_str not in tmp:
                            tmp[annot_str] = set()
                        tmp[annot_str].add(cutoff_int)
                    cutoffs_by_annotation = {
                        annot: sorted(values) for annot, values in tmp.items()
                    }
            except (OSError, ValueError) as err:
                warnings.warn(
                    f"Failed to inspect matches parquet for {key_raw}: {err}",
                )

        payload = {
            "key": key_raw,
            "participants": participants,
            "annotation_ids": annotation_ids,
            "has_matches": has_matches,
            "cutoffs_by_annotation": cutoffs_by_annotation,
        }
        self._send_json(payload)

    def _handle_get_classify_records(self) -> None:
        """Return a page of classification records for a dataset."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        key_raw = (params.get("dataset") or params.get("key") or [""])[0].strip()
        annotation_id = (params.get("annotation_id") or [""])[0].strip()
        participant = (params.get("participant") or ["__all__"])[0].strip()
        page_raw = (params.get("page") or ["1"])[0].strip()
        page_size_raw = (params.get("page_size") or ["25"])[0].strip()
        cutoff_raw = (params.get("score_cutoff") or [""])[0].strip()
        mode_raw = (params.get("score_mode") or [""])[0].strip().lower()

        if not key_raw:
            self._send_json({"error": "Missing dataset key"}, status=400)
            return
        if not annotation_id:
            self._send_json({"error": "Missing annotation_id"}, status=400)
            return
        if "/" in key_raw or "\\" in key_raw:
            self._send_json({"error": "Invalid dataset key"}, status=400)
            return

        try:
            page = int(page_raw)
        except ValueError:
            page = 1
        page = max(page, 1)

        try:
            page_size = int(page_size_raw)
        except ValueError:
            page_size = 25
        page_size = max(page_size, 1)
        page_size = min(page_size, 500)

        score_mode = "eq" if mode_raw == "eq" else "ge"

        root = get_path("annotations_preprocessed").parent
        matches_path = root / f"{key_raw}__matches.parquet"
        preprocessed_path = root / f"{key_raw}__preprocessed.parquet"
        if not preprocessed_path.exists():
            self._send_json({"error": "Dataset not found"}, status=404)
            return

        try:
            # Always load records from the preprocessed per-message table.
            pf = pq.ParquetFile(preprocessed_path)
            col_names = list(pf.schema.names)
            score_column = f"score__{annotation_id}"
            if score_column not in col_names:
                self._send_json(
                    {"error": f"Annotation {annotation_id!r} not present in dataset"},
                    status=400,
                )
                return
            filters: List[tuple[str, str, object]] = []
            if participant and participant != "__all__":
                filters.append(("participant", "=", participant))
            frame = load_annotations_preprocessed_parquet(
                preprocessed_path,
                columns=LOCATION_WITH_CONTEXT_COLUMNS + [score_column],
                filters=filters,
            )
            frame.rename(columns={score_column: "score"}, inplace=True)
            score_cutoff_value: Optional[int]
            if cutoff_raw:
                try:
                    score_cutoff_value = int(cutoff_raw)
                except ValueError:
                    score_cutoff_value = None
            else:
                score_cutoff_value = None
            if score_cutoff_value is not None:
                if score_mode == "eq":
                    frame = frame[frame["score"] == score_cutoff_value]
                else:
                    frame = frame[frame["score"] >= score_cutoff_value]
            frame.insert(0, "annotation_id", annotation_id)
            # No matches information is available in the preprocessed table.
            frame["matches"] = [[] for _ in range(len(frame.index))]
            frame["content"] = ""
            print(
                f"[viewer] classify-records key={key_raw!r} "
                f"annotation_id={annotation_id!r} participant={participant!r} "
                f"cutoff_raw={cutoff_raw!r} mode={score_mode!r} "
                f"rows_after_filters={len(frame.index)}",
            )
        except (OSError, ValueError) as err:
            self._send_json(
                {"error": f"Failed to load records for dataset {key_raw}: {err}"},
                status=500,
            )
            return

        if frame.empty:
            self._send_json(
                {
                    "key": key_raw,
                    "annotation_id": annotation_id,
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "records": [],
                }
            )
            return

        frame.sort_values(
            by=["participant", "source_path", "chat_index", "message_index"],
            inplace=True,
        )

        total = int(frame.shape[0])
        start = (page - 1) * page_size
        if start >= total:
            start = max(0, total - page_size)
        end = min(start + page_size, total)
        page_frame = frame.iloc[start:end].copy()

        # Build a small enrichment mapping from transcripts_data/transcripts.parquet
        # for just the locations on this page so that per-page content loading
        # remains fast.
        enrichment: dict[tuple[str, str, int, int], dict] = {}

        # Prepare filters for looking up any available matches for this
        # annotation/participant combination. The matches parquet is used only
        # to attach validated quote spans and does not drive pagination or
        # score filtering.
        base_filters: List[tuple[str, str, object]] = [
            ("annotation_id", "=", annotation_id)
        ]
        if participant and participant != "__all__":
            base_filters.append(("participant", "=", participant))
        cutoff_for_match: Optional[int] = score_cutoff_value
        if cutoff_for_match is not None:
            filters_for_match = base_filters + [("score_cutoff", "=", cutoff_for_match)]
        else:
            filters_for_match = base_filters

        transcripts_path = get_path("transcripts")
        if transcripts_path.exists() and not page_frame.empty:
            try:
                content_by_key = build_content_mapping_for_locations(
                    transcripts_path,
                    page_frame.to_dict(orient="records"),
                )
                for key_loc, content_value in content_by_key.items():
                    enrichment[key_loc] = {"content": content_value}
            except (OSError, ValueError, TypeError):
                # Best-effort enrichment; fall back to preprocessed rows when
                # transcripts parquet lookups fail.
                enrichment = {}

        if matches_path.exists() and not page_frame.empty:
            try:
                m_frame = load_annotations_matches_parquet(
                    matches_path,
                    columns=[
                        "annotation_id",
                        "participant",
                        "source_path",
                        "chat_index",
                        "message_index",
                        "score_cutoff",
                        "matches",
                    ],
                    filters=filters_for_match,
                )

                # If nothing matched the cutoff-specific filter, fall back to
                # all matches for this annotation (and participant when set).
                if m_frame.empty and cutoff_for_match is not None:
                    m_frame = load_annotations_matches_parquet(
                        matches_path,
                        columns=[
                            "annotation_id",
                            "participant",
                            "source_path",
                            "chat_index",
                            "message_index",
                            "score_cutoff",
                            "matches",
                        ],
                        filters=base_filters,
                    )

                if not m_frame.empty:
                    # Restrict to locations that appear on this page.
                    loc_keys = {
                        (
                            str(row["participant"]),
                            str(row["source_path"]),
                            int(row["chat_index"]),
                            int(row["message_index"]),
                        )
                        for row in page_frame.to_dict(orient="records")
                    }
                    if loc_keys:
                        m_frame = m_frame[
                            m_frame.apply(
                                lambda r: (
                                    str(r["participant"]),
                                    str(r["source_path"]),
                                    int(r["chat_index"]),
                                    int(r["message_index"]),
                                )
                                in loc_keys,
                                axis=1,
                            )
                        ]
                    if not m_frame.empty:
                        # Prefer rows with highest score_cutoff for each location.
                        m_frame.sort_values(
                            by=["score_cutoff"],
                            ascending=[False],
                            inplace=True,
                        )
                        # Keep only the best row per location key.
                        m_frame = m_frame.drop_duplicates(
                            subset=[
                                "participant",
                                "source_path",
                                "chat_index",
                                "message_index",
                            ],
                            keep="first",
                        )
                        for m_row in m_frame.to_dict(orient="records"):
                            key_loc = (
                                str(m_row["participant"]),
                                str(m_row["source_path"]),
                                int(m_row["chat_index"]),
                                int(m_row["message_index"]),
                            )
                            existing = enrichment.get(key_loc, {})
                            combined = dict(existing)
                            raw_matches = m_row.get("matches") or []
                            if isinstance(raw_matches, str):
                                try:
                                    parsed_matches = json.loads(raw_matches)
                                except json.JSONDecodeError:
                                    parsed_matches = []
                                if isinstance(parsed_matches, list):
                                    matches_list = parsed_matches
                                else:
                                    matches_list = []
                            elif isinstance(raw_matches, list):
                                matches_list = raw_matches
                            else:
                                matches_list = []
                            combined["matches"] = matches_list
                            enrichment[key_loc] = combined
            except (OSError, ValueError, TypeError):
                # Best-effort enrichment; fall back to preprocessed rows
                # when matches parquet lookups fail.
                enrichment = {}

        records: List[dict] = []
        for row in page_frame.to_dict(orient="records"):
            key_loc = (
                str(row.get("participant") or ""),
                str(row.get("source_path") or ""),
                int(row.get("chat_index") or 0),
                int(row.get("message_index") or 0),
            )
            enriched = enrichment.get(key_loc, {})

            content_value = enriched.get("content") or ""
            matches_value = enriched.get("matches") or []

            record = {
                "annotation_id": row.get("annotation_id"),
                "participant": row.get("participant"),
                "source_path": row.get("source_path"),
                "chat_index": row.get("chat_index"),
                "message_index": row.get("message_index"),
                "role": row.get("role"),
                "score": row.get("score"),
                "matches": matches_value,
                "content": content_value,
                "timestamp": row.get("timestamp"),
                "chat_key": row.get("chat_key"),
                "chat_date": row.get("chat_date"),
            }
            records.append(record)

        payload = {
            "key": key_raw,
            "annotation_id": annotation_id,
            "total": total,
            "page": page,
            "page_size": page_size,
            "records": records,
        }
        self._send_json(payload)

    def _handle_get_agreement_datasets(self) -> None:
        """Return a list of available agreement datasets."""

        root = Path("analysis") / "agreement"
        if not root.exists():
            self._send_json({"datasets": []})
            return

        datasets: List[dict[str, object]] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            label_timestamp = self._format_manual_dataset_timestamp(None, name)
            label_params = self._format_manual_dataset_params(name)
            label_parts: List[str] = []
            if label_timestamp:
                label_parts.append(label_timestamp)
            if label_params:
                label_parts.append(label_params)
            if label_parts:
                pretty_label = " - ".join(label_parts)
            else:
                pretty_label = name
            try:
                rel = path.relative_to(Path("."))
            except ValueError:
                rel = path
            rel_str = str(rel).replace("\\", "/")
            datasets.append(
                {
                    "key": name,
                    "path": rel_str,
                    "label": pretty_label,
                }
            )

        self._send_json({"datasets": datasets})

    def _handle_get_manual_datasets(self) -> None:
        """Return a list of available manual annotation dataset files."""

        root = Path("manual_annotation_inputs")
        if not root.exists():
            self._send_json({"datasets": []})
            return

        datasets: List[dict[str, object]] = []
        for path in sorted(root.rglob("*.jsonl")):
            try:
                rel = path.relative_to(Path("."))
            except ValueError:
                rel = path
            rel_str = str(rel).replace("\\", "/")
            generated_at: Optional[str] = None
            try:
                first_line = path.open("r", encoding="utf-8").readline()
                meta = json.loads(first_line) if first_line else {}
                if isinstance(meta, dict):
                    value = meta.get("generated_at")
                    if isinstance(value, str) and value.strip():
                        generated_at = value.strip()
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                generated_at = None

            label_timestamp = self._format_manual_dataset_timestamp(
                generated_at, path.name
            )
            label_params = self._format_manual_dataset_params(path.name)
            label_parts: List[str] = []
            if label_timestamp:
                label_parts.append(label_timestamp)
            if label_params:
                label_parts.append(label_params)
            if label_parts:
                pretty_label = " - ".join(label_parts)
            else:
                pretty_label = path.name

            datasets.append(
                {
                    "path": rel_str,
                    "name": path.name,
                    "generated_at": generated_at,
                    "label": pretty_label,
                }
            )

        self._send_json({"datasets": datasets})

    @staticmethod
    def _format_manual_dataset_timestamp(
        generated_at: Optional[str], filename: str
    ) -> Optional[str]:
        """Return a human-readable timestamp for a manual-annotation dataset.

        Parameters
        ----------
        generated_at: Optional[str]
            Optional ISO-formatted timestamp from the dataset metadata.
        filename: str
            Dataset filename, potentially containing a timestamp prefix.

        Returns
        -------
        Optional[str]
            Formatted timestamp (YYYY-MM-DD HH:MM:SS) or None when unavailable.
        """

        if generated_at:
            try:
                dt = datetime.fromisoformat(generated_at)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        base_name = Path(filename).name
        prefix, separator, _ = base_name.partition("__")
        if separator and len(prefix) == 15:
            try:
                dt = datetime.strptime(prefix, "%Y%m%d-%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        return None

    @staticmethod
    def _format_manual_dataset_params(filename: str) -> str:
        """Return a human-readable parameter summary from a dataset filename.

        Parameters
        ----------
        filename: str
            Dataset filename including any encoded parameter fragment.

        Returns
        -------
        str
            Parameters formatted as key=value pairs separated by spaces,
            or an empty string when no parameters can be parsed.
        """

        base_name = Path(filename).name
        stem, _, _ = base_name.partition(".jsonl")
        _, separator, suffix = stem.partition("__")
        if separator and suffix:
            fragment = suffix.strip()
        else:
            fragment = stem.strip()
        if not fragment:
            return ""

        try:
            params = string_to_dict(fragment)
        except (TypeError, ValueError):
            return fragment.replace("&", " ")

        if not params:
            return fragment.replace("&", " ")

        parts = [
            f"{key}={value}"
            for key, value in sorted(params.items(), key=lambda item: str(item[0]))
        ]
        return " ".join(parts)

    def _handle_get_manual_labels(self) -> None:
        """Return any previously saved manual labels for a dataset/annotator."""

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        dataset_path_raw = (params.get("dataset_path") or [""])[0].strip()
        annotator_raw = (params.get("annotator_id") or [""])[0].strip()
        if not dataset_path_raw or not annotator_raw:
            self._send_json(
                {"error": "dataset_path and annotator_id are required"},
                status=400,
            )
            return

        manual_root = Path("manual_annotation_inputs").resolve()
        dataset_file = Path(dataset_path_raw)
        if not dataset_file.is_absolute():
            dataset_file = Path(".").joinpath(dataset_file).resolve()
        try:
            dataset_file.relative_to(manual_root)
        except ValueError:
            self._send_json(
                {
                    "error": "Dataset path must live under manual_annotation_inputs/ for resume."
                },
                status=400,
            )
            return

        safe_annotator = self._normalize_annotator_id(annotator_raw)
        base_name = dataset_file.name
        labels_root = Path("manual_annotation_labels").resolve()
        label_path = labels_root / safe_annotator / base_name
        if not label_path.exists():
            self._send_json({"labels": []})
            return

        latest_by_id: dict[str, dict] = {}
        try:
            for obj in self._iter_label_objects(label_path):
                label_id = obj.get("id")
                label_value = obj.get("label")
                if not label_id or label_value not in {
                    "yes",
                    "no",
                    "not_correctly_formatted",
                }:
                    continue
                latest_by_id[str(label_id)] = obj
        except OSError as err:
            self._send_json({"error": f"Failed to read labels: {err}"}, status=500)
            return

        self._send_json({"labels": list(latest_by_id.values())})

    def _handle_get_manual_instructions(self) -> None:
        """Return shared human-facing instructions for manual annotation."""

        instructions = (
            "You will read one target message at a time and decide whether it "
            "satisfies the selected annotation.\n\n"
            f"{BASE_SCOPE_TEXT}\n\n"
            "Use the annotation description and examples to decide whether the "
            "message matches. When in doubt, err on the side of choosing 'No'."
        )
        self._send_json({"instructions": instructions})

    def _handle_save_manual_labels(self) -> None:
        """Persist manual annotation labels next to the source dataset."""

        try:
            body = self._read_json()
        except ValueError as err:
            self._send_json({"error": str(err)}, status=400)
            return

        dataset_path_raw = str(body.get("dataset_path") or "").strip()
        annotator_raw = str(body.get("annotator_id") or "").strip()
        labels = body.get("labels")
        if not dataset_path_raw:
            self._send_json({"error": "Missing 'dataset_path'"}, status=400)
            return
        if not annotator_raw:
            self._send_json({"error": "Missing 'annotator_id'"}, status=400)
            return
        if not isinstance(labels, list) or not labels:
            self._send_json({"error": "Missing or empty 'labels' list"}, status=400)
            return

        manual_root = Path("manual_annotation_inputs").resolve()
        dataset_file = Path(dataset_path_raw)
        try:
            if not dataset_file.is_absolute():
                dataset_file = Path(".").joinpath(dataset_file).resolve()
        except OSError as err:
            self._send_json({"error": str(err)}, status=400)
            return

        try:
            dataset_file.relative_to(manual_root)
        except ValueError:
            self._send_json(
                {
                    "error": "Dataset path must live under manual_annotation_inputs/ for autosave."
                },
                status=400,
            )
            return

        safe_annotator = self._normalize_annotator_id(annotator_raw)
        base_name = dataset_file.name
        labels_root = Path("manual_annotation_labels").resolve()
        target_dir = labels_root / safe_annotator
        out_path = target_dir / base_name

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            # Merge with any existing labels so updates overwrite prior rows.
            latest_by_id: dict[str, dict] = {}
            if out_path.exists():
                try:
                    for obj in self._iter_label_objects(out_path):
                        label_id = obj.get("id")
                        label_value = obj.get("label")
                        if not label_id or label_value not in {
                            "yes",
                            "no",
                            "not_correctly_formatted",
                        }:
                            continue
                        latest_by_id[str(label_id)] = obj
                except OSError as err:
                    self._send_json(
                        {"error": f"Failed to read existing labels: {err}"}, status=500
                    )
                    return
            for item in labels:
                if not isinstance(item, dict):
                    continue
                label_id = item.get("id")
                label_value = item.get("label")
                if not label_id or label_value not in {
                    "yes",
                    "no",
                    "not_correctly_formatted",
                }:
                    continue
                latest_by_id[str(label_id)] = item
            with out_path.open("w", encoding="utf-8") as handle:
                for obj in latest_by_id.values():
                    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError as err:
            self._send_json({"error": f"Failed to save labels: {err}"}, status=500)
            return

        try:
            rel_path = out_path.resolve().relative_to(Path(".").resolve())
        except ValueError:
            rel_path = out_path

        self._send_json({"ok": True, "path": str(rel_path)})

    def _handle_context_messages(self) -> None:
        """Return neighboring messages for a classified record.

        Expects a JSON body with:

        - ``source_path``: Relative path to the transcript JSON recorded in the
          Parquet exports.
        - ``participant``: Optional participant bucket identifier.
        - ``chat_index``: Zero-based conversation index used during
          classification.
        - ``message_index``: Zero-based message index within the conversation.
        - ``depth``: Optional maximum number of messages to return before and
          after the target message.
        """

        try:
            body = self._read_json()
        except ValueError as err:
            self._send_json({"error": str(err)}, status=400)
            return

        source_path = str(body.get("source_path") or "").strip()
        if not source_path:
            self._send_json({"error": "Missing 'source_path'"}, status=400)
            return

        try:
            chat_index_raw = body.get("chat_index")
            message_index_raw = body.get("message_index")
            chat_index = int(chat_index_raw)
            message_index = int(message_index_raw)
        except (TypeError, ValueError):
            self._send_json(
                {"error": "chat_index and message_index must be integers"},
                status=400,
            )
            return

        depth_raw = body.get("depth")
        try:
            depth = int(depth_raw) if depth_raw is not None else 3
        except (TypeError, ValueError):
            self._send_json({"error": "depth must be an integer"}, status=400)
            return

        participant = str(body.get("participant") or "").strip() or None

        try:
            previous, next_messages = load_context_messages(
                source_path,
                chat_index,
                message_index,
                depth,
                participant=participant,
            )
        except (FileNotFoundError, OSError, ValueError) as err:
            self._send_json({"error": str(err)}, status=400)
            return

        self._send_json({"previous": previous, "next": next_messages})


def _load_transcript_messages_from_parquet(
    transcripts_path: Path,
    source_path: str,
    chat_index: int,
    participant: str | None,
) -> List[dict]:
    """Return simplified messages for a chat from transcripts Parquet data.

    Parameters
    ----------
    transcripts_path:
        Path to ``transcripts.parquet`` with message content.
    source_path:
        Relative transcript path recorded in Parquet exports.
    chat_index:
        Zero-based conversation index within the transcript file.
    participant:
        Optional participant bucket identifier to disambiguate transcripts.

    Returns
    -------
    List[dict]
        Simplified message dictionaries with ``index``, ``role``, ``content``,
        and ``timestamp`` keys.
    """

    resolved = transcripts_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Transcripts parquet not found: {resolved}")

    filters: List[tuple[str, str, object]] = [
        ("source_path", "=", source_path),
        ("chat_index", "=", chat_index),
    ]
    if participant:
        filters.append(("participant", "=", participant))

    try:
        frame = load_transcripts_parquet(
            resolved,
            columns=[
                "participant",
                "source_path",
                "chat_index",
                "message_index",
                "role",
                "timestamp",
                "content",
            ],
            filters=filters,
        )
    except (OSError, ValueError, TypeError) as err:
        raise OSError(f"Failed to read transcripts parquet: {err}") from err

    if frame.empty:
        raise ValueError(
            "No transcript rows found for "
            f"source_path={source_path!r}, chat_index={chat_index}"
        )

    if not participant:
        participants = sorted(
            {
                str(value).strip()
                for value in frame["participant"].dropna().tolist()
                if str(value).strip()
            }
        )
        if len(participants) > 1:
            raise ValueError(
                "Multiple participants found for "
                f"source_path={source_path!r}, chat_index={chat_index}; "
                "provide participant to disambiguate"
            )

    frame.sort_values(by=["message_index"], inplace=True)

    messages: List[dict] = []
    for row in frame.to_dict(orient="records"):
        try:
            msg_index = int(row.get("message_index"))
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid message_index in transcripts parquet: {err}")

        role = str(row.get("role") or "")
        content = row.get("content")
        text = "" if content is None else str(content)
        timestamp = normalize_timestamp_value(row.get("timestamp")) or ""
        messages.append(
            {
                "index": msg_index,
                "role": role,
                "content": text,
                "timestamp": timestamp,
            }
        )

    if not messages:
        raise ValueError(
            "No usable messages found for "
            f"source_path={source_path!r}, chat_index={chat_index}"
        )

    return messages


def load_context_messages(
    source_path: str,
    chat_index: int,
    message_index: int,
    depth: int,
    *,
    participant: str | None = None,
) -> Tuple[List[dict], List[dict]]:
    """Return messages before and after a target message using Parquet data.

    Parameters
    ----------
    source_path:
        Relative path recorded in the Parquet exports (``source_path`` field).
    chat_index:
        Zero-based index of the conversation within the file.
    message_index:
        Zero-based index of the target message within the conversation.
    depth:
        Maximum number of messages to include before and after the target.
    participant:
        Optional participant bucket identifier to disambiguate transcripts.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        Two lists of simplified message dicts: ``(previous, next)``.
    """

    transcripts_path = get_path("transcripts")
    messages = _load_transcript_messages_from_parquet(
        transcripts_path,
        source_path,
        chat_index,
        participant,
    )

    depth_limit = depth if depth > 0 else 0
    depth_limit = min(depth_limit, 10)

    index_map: dict[int, int] = {}
    for idx, msg in enumerate(messages):
        msg_index = msg.get("index")
        if isinstance(msg_index, int):
            index_map[msg_index] = idx

    if message_index not in index_map:
        raise ValueError(
            f"message_index {message_index} not found for "
            f"source_path={source_path!r}, chat_index={chat_index}"
        )

    target_idx = index_map[message_index]

    previous_indices = compute_previous_indices_skipping_roles(
        messages,
        target_idx,
        depth_limit,
        skip_roles=("tool",),
    )
    next_start = target_idx + 1
    next_end = min(len(messages), next_start + depth_limit)
    next_range = range(next_start, next_end)

    previous = [messages[i] for i in previous_indices]
    next_messages = [messages[i] for i in next_range]
    return previous, next_messages


def parse_args() -> argparse.Namespace:
    """Return command-line arguments for HTTP server configuration."""
    parser = argparse.ArgumentParser(
        description="Start a no-cache HTTP server for development assets."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory to serve (defaults to current directory).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind against (defaults to 8000).",
    )
    return parser.parse_args()


def serve(directory: Path, port: int) -> None:
    """Start the HTTP server bound to the requested directory and port."""
    handler = partial(NoCacheRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("localhost", port), handler)

    print(
        f"Serving {directory.resolve()} on http://localhost:{port}/ with caching disabled."
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    """Entry point that starts the no-cache HTTP server."""
    # Suppress noisy multiprocessing resource_tracker warnings on Ctrl+C shutdown
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be .* leaked semaphore objects",
        category=UserWarning,
    )
    args = parse_args()
    serve(directory=args.directory, port=args.port)


# Install camel-case HTTP method alias after class definition to satisfy http.server
# while keeping a snake_case implementation for linting.
NoCacheRequestHandler.do_POST = NoCacheRequestHandler.do_post
_ = NoCacheRequestHandler.do_POST


def load_classify_chats() -> ModuleType:
    """Load the ``scripts/annotation/classify_chats.py`` module.

    Returns
    -------
    ModuleType
        Loaded module exposing ``parse_args`` and ``main``.
    """
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "scripts" / "annotation" / "classify_chats.py"
    module_name = "scripts.annotation.classify_chats"
    scripts_dir = str((project_root / "scripts").resolve())
    # Ensure the scripts package root is importable so annotation modules resolve
    added_path = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added_path = True

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to locate classify_chats at {module_path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert hasattr(loader, "exec_module")  # narrow type for mypy/pylint
    # Ensure fresh annotations on each load so edits in annotations.csv are honored
    if "annotations" in sys.modules:
        del sys.modules["annotations"]
    # Register in sys.modules so decorators (e.g., dataclasses) can resolve __module__
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    finally:
        # Leave scripts_dir on sys.path so subsequent imports behave consistently
        # (e.g., later classify runs). Do not remove to avoid race conditions.
        if added_path:
            pass
    return module


if __name__ == "__main__":
    main()
