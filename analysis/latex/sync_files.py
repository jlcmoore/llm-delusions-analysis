"""Synchronize analysis assets from the llm-delusions repo into the
llm-delusions-overleaf LaTeX repo.

This script copies CSV and figure outputs from this analysis repository
into the sibling LaTeX repository and generates simple LaTeX ``tabular``
files for selected tables. It is intended to be rerun whenever the
upstream analyses are updated so that tables and figures stay in sync
without manual data editing.

The expected directory layout is:

- this file: llm-delusions/analysis/latex/sync_files.py
- LaTeX repo: ../llm-delusions-overleaf

Run from anywhere (paths are resolved relative to this file), for example:

    python analysis/latex/sync_files.py
"""

from __future__ import annotations

import pathlib
import re
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd
from llm_delusions_annotations.annotation_metadata import (
    EXCLUDED_ANNOTATION_IDS,
    ID_COLUMN,
    normalize_scope,
)
from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    ANNOTATION_TEMPLATE,
    ANNOTATIONS_FILE,
    BASE_SCOPE_TEXT,
)

from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.reporting.latex_escape import escape_latex
from analysis.lib.reporting.latex_tables import csv_to_latex_tabular
from annotation.agreement_columns import (
    inter_annotator_annotation_columns,
    majority_annotation_columns,
)
from utils.participants import normalize_participant_id

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
LLM_DELUSIONS_ROOT = ANALYSIS_ROOT.parent
LATEX_ROOT = LLM_DELUSIONS_ROOT.parent / "llm-delusions-overleaf"


def _ensure_dir(path: pathlib.Path) -> None:
    """Create directory ``path`` if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Copy a file from ``src`` to ``dest``, creating parents as needed."""
    if not src.exists():
        raise FileNotFoundError(f"Expected source asset not found: {src}")
    _ensure_dir(dest.parent)
    shutil.copy2(src, dest)


def _write_annotation_prompt_snippet() -> None:
    """Write the combined annotation system prompt and template using listings.

    The output is intended to be included in an ACM-style figure environment
    in the LaTeX repository. The content is taken directly from the shared
    constants so that any future edits are reflected automatically.
    """

    figures_dir = LATEX_ROOT / "figures"
    _ensure_dir(figures_dir)
    dest = figures_dir / "annotation_prompt_snippet.tex"

    template_with_scope = ANNOTATION_TEMPLATE.replace(
        "{base_scope_text}", BASE_SCOPE_TEXT
    )
    combined = ANNOTATION_SYSTEM_PROMPT + "\n\n" + template_with_scope
    combined_ascii = combined.encode("ascii", "ignore").decode("ascii")

    with dest.open("w", encoding="utf-8") as handle:
        handle.write("% Annotation system prompt and template\n")
        handle.write(
            "\\lstset{\n"
            "      basicstyle=\\ttfamily\\small,\n"
            "      breaklines=true,\n"
            "      breakatwhitespace=true,\n"
            "      columns=fullflexible,\n"
            "      showstringspaces=false,\n"
            "      keepspaces=true,\n"
            "      keywordstyle=\\ttfamily\\small,\n"
            "      commentstyle=\\ttfamily\\small,\n"
            "      stringstyle=\\ttfamily\\small,\n"
            "      language={}\n"
            "}\n"
        )
        handle.write("\\begin{lstlisting}\n")
        handle.write(combined_ascii)
        handle.write("\n\\end{lstlisting}\n")


def _escape_latex_annotation(text):
    """Escape special LaTeX characters for annotation fields."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove problematic emoji and other non-ASCII characters.
    text = text.encode("ascii", "ignore").decode("ascii")

    return escape_latex(text)


def _clean_example_text(text):
    """Clean example text for LaTeX table."""
    if pd.isna(text):
        return ""

    text = str(text)
    # Remove line numbers and formatting artifacts.
    text = re.sub(r"^\d+→", "", text, flags=re.MULTILINE)
    # Replace multiple newlines with single newline.
    text = re.sub(r"\n\s*\n+", r"\n", text)
    # Trim whitespace.
    text = text.strip()

    # Remove problematic emoji and other non-ASCII characters.
    text = text.encode("ascii", "ignore").decode("ascii")

    return escape_latex(text)


def _resolve_output_path(output_path: str) -> Path:
    """Resolve the output path to an absolute filesystem path."""
    return Path(output_path).expanduser().resolve()


def _load_and_filter_annotations(annotations_path: Path, required_cols):
    """Load annotations CSV and filter required, non-empty rows."""
    df = pd.read_csv(annotations_path)

    # Always include the identifier column so that exclusion lists can be applied.
    all_required = list(dict.fromkeys(list(required_cols) + [ID_COLUMN]))

    missing = [col for col in all_required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_filtered = df[all_required].copy()
    df_filtered = df_filtered[df_filtered["name"].notna()]

    return df_filtered


def _write_annotation_header(file_handle) -> None:
    """Write the static LaTeX header for the annotation schema description list."""
    file_handle.write("% Annotation schema description list\n\n")
    file_handle.write("\\begin{description}[style=nextline,leftmargin=0pt]\n")


def _truncate_examples(examples: str) -> str:
    """Truncate example text to only the first line."""
    if not examples:
        return ""

    lines = examples.split("\n")
    return lines[0]


def _format_scope(scope: str) -> str:
    """Format annotation scope for LaTeX output."""
    cleaned = scope.strip()
    if cleaned in ["user, assistant", "user,assistant"]:
        return "user, chatbot"
    if cleaned == "assistant":
        return "chatbot"
    return cleaned


def _write_annotation_table_rows(df_filtered, file_handle) -> None:
    """Write all annotation entries into a LaTeX description list."""
    for _, row in df_filtered.iterrows():
        annotation_id = str(row.get(ID_COLUMN, "")).strip()
        if annotation_id and annotation_id in EXCLUDED_ANNOTATION_IDS:
            continue

        name = _escape_latex_annotation(row["name"])
        description = _escape_latex_annotation(row["description"])
        pos_ex = _clean_example_text(row["positive-examples"])
        neg_ex = _clean_example_text(row["negative-examples"])

        if description == "":
            continue

        raw_scope_value = row.get("scope", "")
        if pd.isna(raw_scope_value):
            raw_scope = ""
        else:
            raw_scope = str(raw_scope_value)

        # When an annotation applies to both user and assistant, emit two
        # separate entries so that no row appears scoped to both roles.
        scope_roles = normalize_scope(raw_scope)
        if "user" in scope_roles and "assistant" in scope_roles:
            scope_variants = ["user", "assistant"]
        elif scope_roles:
            scope_variants = [",".join(scope_roles)]
        else:
            scope_variants = [raw_scope]

        description_text = description.replace("\n", " ")
        positive_example = _truncate_examples(pos_ex)
        negative_example = _truncate_examples(neg_ex)

        for scope_variant in scope_variants:
            scope_text = _format_scope(scope_variant)

            file_handle.write(f"\\item[{name}]%\n")
            if scope_text:
                file_handle.write(f"\\textbf{{Scope:}} {scope_text}\\\\\n")
            if description_text:
                file_handle.write(
                    f"\\textbf{{Description:}} {description_text}\\\\\n",
                )
            if positive_example:
                file_handle.write(
                    f"\\textbf{{Positive example:}} {positive_example}\\\\\n",
                )
            if negative_example:
                file_handle.write(
                    f"\\textbf{{Negative example:}} {negative_example}\\\\\n",
                )


def _create_annotation_table(annotations_path: Path, output_path: Path) -> None:
    """Create the LaTeX annotation schema table from the CSV file."""
    required_cols = [
        "name",
        "scope",
        "description",
        "positive-examples",
        "negative-examples",
    ]

    df_filtered = _load_and_filter_annotations(annotations_path, required_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file_handle:
        _write_annotation_header(file_handle)
        _write_annotation_table_rows(df_filtered, file_handle)
        file_handle.write("\\end{description}\n")


def build_annotation_table(output_path: str | None = None) -> Path:
    """Build the annotation schema table for programmatic callers.

    Parameters
    ----------
    output_path:
        Optional filesystem path where the LaTeX file should be written.
        When omitted, the table is written into the LaTeX repo at
        ``<LATEX_ROOT>/tables/annotation_schema.tex``.

    Returns
    -------
    pathlib.Path
        Absolute path to the generated LaTeX file.
    """

    annotations_path = Path(ANNOTATIONS_FILE)
    if output_path is None:
        tables_dir = LATEX_ROOT / "tables"
        _ensure_dir(tables_dir)
        resolved_output_path = tables_dir / "annotation_schema.tex"
    else:
        resolved_output_path = _resolve_output_path(output_path)
    _create_annotation_table(annotations_path, resolved_output_path)
    return resolved_output_path


def _transform_annotation_frequency_row(row: Dict[str, str]) -> Dict[str, str]:
    """Return a LaTeX-friendly view of a single annotation frequency row."""
    transformed = dict(row)

    # Shorten annotation id for more compact display, reusing the shared
    # label helper used by plotting code. Keep the identifier on a single
    # line for this table.
    annotation_id = transformed.get("annotation_id", "")
    if annotation_id:
        short_id = shorten_annotation_label(annotation_id)
        # Replace underscores so we do not need to escape them in LaTeX.
        short_id = short_id.replace("_", "-")
        transformed["annotation_id"] = short_id

    # Shorten scope role labels.
    scope_value = transformed.get("scope", "")
    if scope_value:
        scope_value = scope_value.replace("assistant", "ast").replace("user", "usr")
        transformed["scope"] = scope_value

    return transformed


def _transform_participant_stats_row(row: Dict[str, str]) -> Dict[str, str]:
    """Return a LaTeX-friendly view of a participant stats row."""
    transformed = dict(row)
    participant_value = str(transformed.get("participant", "")).strip()

    # Hide the participant ID for all rows except the summary TOTAL row.
    if participant_value.upper() == "TOTAL":
        transformed["participant"] = "TOTAL"
    else:
        transformed["participant"] = ""

    models_value = transformed.get("models", "")
    if models_value:
        # Ensure a space after each semicolon so models wrap more readably.
        transformed["models"] = models_value.replace(";", "; ")
    return transformed


def sync_tables() -> None:
    """Sync key CSV tables and generate LaTeX tabular files."""
    tables_dir = LATEX_ROOT / "tables"
    _ensure_dir(tables_dir)

    # Descriptive stats on transcripts
    transcript_csv = (
        LLM_DELUSIONS_ROOT / "analysis" / "data" / "participant_transcript_stats.csv"
    )
    if transcript_csv.exists():
        transcript_tex = tables_dir / "participant_transcript_stats.tex"
        csv_to_latex_tabular(
            transcript_csv,
            transcript_tex,
            columns=[
                "participant",
                "gender",
                "age_bin",
                "num_conversations",
                "total_messages",
                "max_conversation_length",
                "median_conversation_length",
                "longest_conversation_rate",
                "files",
                "top_model",
                "top_model_rate",
                "span_days",
            ],
            header_labels={
                "participant": "participant",
                "gender": "gender",
                "age_bin": "age",
                "num_conversations": r"\shortstack{n\\conv.}",
                "total_messages": r"\shortstack{n\\msg.s}",
                "max_conversation_length": r"\shortstack{max\\conv. len.}",
                "median_conversation_length": r"\shortstack{median\\conv. len.}",
                "longest_conversation_rate": r"\shortstack{when\\longest conv.}",
                "files": "files",
                "top_model": r"\shortstack{top\\chatbot}",
                "top_model_rate": r"\shortstack{pr. top\\chatbot}",
                "span_days": r"\shortstack{span\\days}",
            },
            raw_header_columns={
                "num_conversations",
                "total_messages",
                "max_conversation_length",
                "median_conversation_length",
                "longest_conversation_rate",
                "top_model",
                "top_model_rate",
                "span_days",
            },
            row_transform=_transform_participant_stats_row,
            # Make the models column a wrapped paragraph column.
        )
    else:
        print(f"[sync] Skipping missing CSV: {transcript_csv}")

    # Global annotation frequencies (score cutoff = 0)
    annotation_freq_csv = (
        LLM_DELUSIONS_ROOT / "analysis" / "data" / "annotation_frequencies.csv"
    )
    if annotation_freq_csv.exists():
        annotation_freq_tex = tables_dir / "annotation_frequencies.tex"
        csv_to_latex_tabular(
            annotation_freq_csv,
            annotation_freq_tex,
            columns=[
                "annotation_id",
                "category",
                "n_positive_all",
                "rate_participants_mean",
                "rate_participants_positive",
            ],
            header_labels={
                "annotation_id": "Annotation id",
                "category": "Category",
                "n_positive_all": "n pos.",
                "rate_participants_mean": "pr. ppt mean",
                "rate_participants_positive": "pr. ppts. (> 4 msgs)",
            },
            row_transform=_transform_annotation_frequency_row,
            raw_columns={"annotation_id"},
            category_collapse_column="category",
            group_break_column="category",
        )
    else:
        print(f"[sync] Skipping missing CSV: {annotation_freq_csv}")

    # Agreement summary tables (compact CSVs for majority and inter-annotator)
    majority_csv = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "latex"
        / "generated"
        / "agreement_summary_majority_compact.csv"
    )
    if majority_csv.exists():
        majority_tex = tables_dir / "agreement_summary_majority_compact.tex"
        csv_to_latex_tabular(majority_csv, majority_tex)
    else:
        print(f"[sync] Skipping missing CSV: {majority_csv}")

    inter_annotator_csv = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "latex"
        / "generated"
        / "agreement_summary_inter_annotator_compact.csv"
    )
    if inter_annotator_csv.exists():
        inter_annotator_tex = (
            tables_dir / "agreement_summary_inter_annotator_compact.tex"
        )
        csv_to_latex_tabular(inter_annotator_csv, inter_annotator_tex)
    else:
        print(f"[sync] Skipping missing CSV: {inter_annotator_csv}")

    # Per-annotation agreement tables for the test dataset (majority and
    # human inter-annotator), excluding the dataset column in the LaTeX
    # versions so that rows are keyed by annotation id only.
    majority_by_annotation_csv = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "latex"
        / "generated"
        / "agreement_summary_majority_by_annotation_test.csv"
    )
    if majority_by_annotation_csv.exists():
        majority_by_annotation_tex = (
            tables_dir / "agreement_summary_majority_by_annotation_test.tex"
        )
        csv_to_latex_tabular(
            majority_by_annotation_csv,
            majority_by_annotation_tex,
            columns=[
                "annotation_id",
                "score_cutoff",
                "items",
                "tp",
                "fp",
                "tn",
                "fn",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "kappa",
            ],
            header_labels={
                "annotation_id": "code",
                "score_cutoff": "cutoff",
            },
            col_spec=r"p{0.35\linewidth}" + "r" * 11,
        )
    else:
        print(f"[sync] Skipping missing CSV: {majority_by_annotation_csv}")

    inter_annotator_by_annotation_csv = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "latex"
        / "generated"
        / "agreement_summary_inter_annotator_by_annotation_test.csv"
    )
    if inter_annotator_by_annotation_csv.exists():
        inter_annotator_by_annotation_tex = (
            tables_dir / "agreement_summary_inter_annotator_by_annotation_test.tex"
        )
        csv_to_latex_tabular(
            inter_annotator_by_annotation_csv,
            inter_annotator_by_annotation_tex,
            columns=[
                "annotation_id",
                "score_cutoff",
                "items",
                "pos_agree",
                "neg_agree",
                "pos_disagree",
                "neg_disagree",
                "agreement_rate",
                "kappa",
            ],
            header_labels={
                "annotation_id": "code",
                "score_cutoff": "cutoff",
                "agreement_rate": "agreement",
                "pos_agree": "pos agree",
                "neg_agree": "neg agree",
                "pos_disagree": "pos dis",
                "neg_disagree": "neg dis",
            },
            col_spec=r"p{0.35\linewidth}" + "r" * 8,
        )
    else:
        print(f"[sync] Skipping missing CSV: {inter_annotator_by_annotation_csv}")


def sync_figures() -> None:
    """Sync key figure files (PR curves and length effects)."""
    figures_dir = LATEX_ROOT / "figures"
    _ensure_dir(figures_dir)

    # Agreement precision-recall curves (validation)
    pr_overall = (
        LLM_DELUSIONS_ROOT / "analysis" / "figures" / "agreement" / "pr__validation.pdf"
    )
    if pr_overall.exists():
        pr_overall_dest = figures_dir / "agreement_pr__validation.pdf"
        _copy_file(pr_overall, pr_overall_dest)
    else:
        print(f"[sync] Skipping missing figure: {pr_overall}")

    pr_by_annotation = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "agreement"
        / "pr_by_annotation__validation__gpt-5.1.pdf"
    )
    if pr_by_annotation.exists():
        pr_by_annotation_dest = (
            figures_dir / "agreement_pr_by_annotation__validation__gpt-5.1.pdf"
        )
        _copy_file(pr_by_annotation, pr_by_annotation_dest)
    else:
        print(f"[sync] Skipping missing figure: {pr_by_annotation}")

    # Conversation-length effects on annotation prevalence (remaining-length regression)
    remaining_length_effects = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "annotation_remaining_length_histogram.pdf"
    )
    if remaining_length_effects.exists():
        remaining_length_effects_dest = (
            figures_dir / "annotation_remaining_length_histogram.pdf"
        )
        _copy_file(remaining_length_effects, remaining_length_effects_dest)
    else:
        print(f"[sync] Skipping missing figure: {remaining_length_effects}")

    remaining_length_effects_extremes = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "annotation_remaining_length_histogram_extremes.pdf"
    )
    if remaining_length_effects_extremes.exists():
        remaining_length_effects_extremes_dest = (
            figures_dir / "annotation_remaining_length_histogram_extremes.pdf"
        )
        _copy_file(
            remaining_length_effects_extremes, remaining_length_effects_extremes_dest
        )
    else:
        print(f"[sync] Skipping missing figure: {remaining_length_effects_extremes}")

    # Annotation frequency histogram (participant-normalized rates)
    freq_histogram = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "annotation_frequency_histogram.pdf"
    )
    if freq_histogram.exists():
        freq_histogram_dest = figures_dir / "annotation_frequency_histogram.pdf"
        _copy_file(freq_histogram, freq_histogram_dest)
    else:
        print(f"[sync] Skipping missing figure: {freq_histogram}")

    # Annotation frequency histogram by category (participant-normalized rates)
    freq_histogram_categories = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "annotation_frequency_histogram_categories.pdf"
    )
    if freq_histogram_categories.exists():
        freq_histogram_categories_dest = (
            figures_dir / "annotation_frequency_histogram_categories.pdf"
        )
        _copy_file(freq_histogram_categories, freq_histogram_categories_dest)
    else:
        print(f"[sync] Skipping missing figure: {freq_histogram_categories}")

    # Annotation-set frequency histogram (participant-normalized set rates)
    set_freq_histogram = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "annotation_set_frequency_histogram.pdf"
    )
    if set_freq_histogram.exists():
        set_freq_histogram_dest = figures_dir / "annotation_set_frequency_histogram.pdf"
        _copy_file(set_freq_histogram, set_freq_histogram_dest)
    else:
        print(f"[sync] Skipping missing figure: {set_freq_histogram}")

    # Annotation-set frequency histogram comparison by model
    set_freq_histogram_compare = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "annotation_set_frequency_histogram_compare.pdf"
    )
    if set_freq_histogram_compare.exists():
        set_freq_histogram_compare_dest = (
            figures_dir / "annotation_set_frequency_histogram_compare.pdf"
        )
        _copy_file(set_freq_histogram_compare, set_freq_histogram_compare_dest)
    else:
        print(f"[sync] Skipping missing figure: {set_freq_histogram_compare}")

    # Sequential dynamics enrichment summary
    enrichment_ks = (
        LLM_DELUSIONS_ROOT / "analysis" / "figures" / "sequential_enrichment_Ks.pdf"
    )
    if enrichment_ks.exists():
        enrichment_ks_dest = figures_dir / "sequential_enrichment_Ks.pdf"
        _copy_file(enrichment_ks, enrichment_ks_dest)
    else:
        print(f"[sync] Skipping missing figure: {enrichment_ks}")

    # Sequential dynamics paired profiles
    suicidal_violent = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "sequential_profile_suicidal_and_violent.pdf"
    )
    if suicidal_violent.exists():
        suicidal_violent_dest = (
            figures_dir / "sequential_profile_suicidal_and_violent.pdf"
        )
        _copy_file(suicidal_violent, suicidal_violent_dest)
    else:
        print(f"[sync] Skipping missing figure: {suicidal_violent}")

    romantic_personhood = (
        LLM_DELUSIONS_ROOT
        / "analysis"
        / "figures"
        / "sequential_profile_romantic_and_personhood.pdf"
    )
    if romantic_personhood.exists():
        romantic_personhood_dest = (
            figures_dir / "sequential_profile_romantic_and_personhood.pdf"
        )
        _copy_file(romantic_personhood, romantic_personhood_dest)
    else:
        print(f"[sync] Skipping missing figure: {romantic_personhood}")

    _write_annotation_prompt_snippet()


def main() -> None:
    """Run all synchronization steps."""
    if not LATEX_ROOT.exists():
        raise FileNotFoundError(
            f"Expected LaTeX repo at {LATEX_ROOT}, " "but the directory does not exist."
        )
    print("[sync] Building annotation schema table...")
    build_annotation_table()
    sync_tables()
    sync_figures()


if __name__ == "__main__":
    main()
