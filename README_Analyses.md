# Analyses

## Contents

- [Preprocessing](#preprocessing)
- [Transcript tables for analysis](#transcript-tables-for-analysis)
- [Participant demographics from surveys](#participant-demographics-from-surveys)
- [Participant ordering and transcript stats](#participant-ordering-and-transcript-stats)
- [Global annotation frequency table](#global-annotation-frequency-table)
- [Per-participant annotation trajectories (time and sequence)](#per-participant-annotation-trajectories-time-and-sequence)
- [Per-participant stacked annotation streamgraphs](#per-participant-stacked-annotation-streamgraphs)
- [Conversation counts (currently unused)](#conversation-counts-currently-unused)
- [Annotation profiles](#annotation-profiles)
- [Sequential dynamics](#sequential-dynamics)
- [Topic-modeling inputs from annotation matches](#topic-modeling-inputs-from-annotation-matches)
- [Optional topic modeling analyses](#optional-topic-modeling-analyses)

## Preprocessing

Note: Analysis scripts now import shared helpers from `analysis.lib`. Run them as
modules (for example, `python -m analysis.compute_annotation_frequencies`) so the
`analysis` package is on `PYTHONPATH`.

### Annotations

We build a parquet file from the annotations for easy lookup

```bash
python -m analysis.preprocess_annotation_family \
  annotation_outputs/101/all_annotations__part-0001.jsonl \
  --with-matches # Optional, generates both preprocessed and matches tables \
  --output annotations/all_annotations__preprocessed.parquet
```

By default, this normalizes legacy `source_path` values (for example,
`human_line/hl_03/...` or `under_irb/irb_01/...`) into numeric participant
layouts (`203/...`, `101/...`) so joins against `transcripts_data` work
consistently.

### Transcript tables for analysis

To build columnar tables for all transcript messages (separate from annotations), run:

```bash
python scripts/parse/export_transcripts_parquet.py \
  --transcripts-root transcripts_de_ided \
  --output-dir transcripts_data
```

This writes:

- `transcripts_data/transcripts_index.parquet` – one row per message with keys and lightweight metadata:
  - `participant`, `source_path`, `chat_index`, `message_index`, `role`, `timestamp`, `chat_key`, `chat_date`.
- `transcripts_data/transcripts.parquet` – the same rows plus a `content` column.

These tables are designed to be joined with annotation tables on the shared location key (`participant`, `source_path`, `chat_index`, `message_index`) without repeatedly reparsing the raw JSON under `transcripts_de_ided/`.

#### Example: join annotations to transcript index (no content)

With the preprocessed annotations and transcript index in place, you can do a fast, content-free join in a notebook or script:

```python
import pandas as pd
from pathlib import Path

annotations_path = Path("annotations/all_annotations__preprocessed.parquet")
index_path = Path("transcripts_data/transcripts_index.parquet")

annotations = pd.read_parquet(annotations_path)
transcript_index = pd.read_parquet(index_path)

merged = annotations.merge(
    transcript_index,
    on=["participant", "source_path", "chat_index", "message_index"],
    how="inner",
    suffixes=("_ann", "_tx"),
)
```

`merged` now contains per-message annotation scores plus transcript-level metadata (role, timestamp, conversation key/date) without loading any `content`.

#### Example: attach content for a specific message

When you need the full text for a small subset of messages, you can look up `content` on demand from `transcripts.parquet`:

```python
from pathlib import Path

transcripts_path = Path("transcripts_data/transcripts.parquet")

# Pick one joined row as an example.
row = merged.iloc[0]

filters = [
    ("participant", "=", row["participant"]),
    ("source_path", "=", row["source_path"]),
    ("chat_index", "=", int(row["chat_index"])),
    ("message_index", "=", int(row["message_index"])),
]

transcript_row = pd.read_parquet(
    transcripts_path,
    engine="pyarrow",
    filters=filters,
)

print(transcript_row["content"].iloc[0])
```

This pattern keeps most analysis on the lightweight index and annotations tables, only touching `content` for the specific messages you need.

<!--  -->
<!--  -->
<!--  -->
<!--  -->

## Participant demographics from surveys

To compute basic participant demographics (age and gender) from the IRB survey JSON files in `surveys/` and write a CSV suitable for paper tables, run:

```bash
python -m analysis.compute_demographics \
  --surveys-dir surveys \
  --output analysis/data/demographics.csv
```

This script:

- Scans `1*.json` files in the specified surveys directory (by default `surveys/`), using the keys `"What is your age?"` and `"What is your gender? - Selected Choice"`.
- Prints a text summary of age range, mean, median, and gender breakdown to the console.
- Writes a long-format CSV to `analysis/data/demographics.csv` with fields, categories, counts, and percentages that can be dropped directly into a LaTeX table.

## Participant ordering and transcript stats

To compute per-participant ordering categories (for downstream sequential analyses) and high-level transcript statistics that do not depend on annotation outputs, run:

```bash
python scripts/parse/compute_participant_ordering_and_stats.py \
 --transcripts-root transcripts_de_ided \
 --ordering-json analysis/participant_ordering.json \
 --stats-csv analysis/data/participant_transcript_stats.csv
```

This scans `transcripts_de_ided` using the same linearized, visible user/assistant message paths as the rest of the pipeline and writes:

- `analysis/participant_ordering.json`: ordering category per participant (for example, whether a global ordering over messages is available).
- `analysis/data/participant_transcript_stats.csv`: per-participant summary table (conversation and message counts, lengths, file types, and model usage) sorted by total messages with a final `TOTAL` row.

### By participant plots

We currently have static analyses available to run on the de-ided transcripts. These produces various graphs in `analysis/figures` for each of the participants passed (look at the subdirectories).

Example analysis artifacts:

- [conversation_counts.png](analysis/figures/conversation_counts.png) summarises the total chats per participant.
- [105 sequence summary](analysis/figures/participants/105/105_message_level_summary.png) shows message mix for a single participant.

<img src="analysis/figures/conversation_counts.png" alt="Conversation counts bar chart" width="320" />

```bash
python -m analysis.make_participant_plots transcripts_de_ided/
```

To run just the analyses for one participant:

```bash
python -m analysis.make_participant_plots transcripts_de_ided/ --participants 105
```

## Global annotation frequency table

To compute the global and marginal annotation frequencies used in the paper, run:

```bash
python -m analysis.compute_annotation_frequencies \
    annotations/all_annotations__preprocessed.parquet \
    --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
    --long-quantile 0.75 \
    --output analysis/data/annotation_frequencies.csv
```

This command:

- First materialises a per-message Parquet table (`annotations/all_annotations__preprocessed.parquet`) that aggregates scores across the full job family discovered from the reference JSONL.
- Then applies per-annotation score cutoffs from the cutoffs CSV file to compute message-pooled, participant-averaged, and length-enrichment statistics for each annotation.
- Writes the canonical frequency table to `analysis/data/annotation_frequencies.csv`.

To plot the per-category histogram with set-frequency overlays:

```bash
python analysis/plot_annotation_frequency_histogram_categories.py \
  analysis/data/annotation_frequencies.csv \
  --set-frequency-csv analysis/data/annotation_set_frequencies__by_model.csv \
  --output analysis/figures/annotation_frequency_histogram_categories.pdf
```

### By-model annotation-set frequencies

To compute annotation-set (category) prevalences by model id (plus an overall
row across all messages), run:

```bash
python analysis/compute_annotation_set_frequencies.py \
  annotations/all_annotations__preprocessed.parquet \
  --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
  --llm-score-cutoff 5 \
  --transcripts-parquet transcripts_data/transcripts.parquet \
  --model gpt-4o \
  --model gpt-5 \
  --output analysis/data/annotation_set_frequencies__by_model.csv
```

To plot the comparison histogram directly from that CSV:

```bash
python analysis/plot_annotation_set_frequencies_compare.py \
  --input analysis/data/annotation_set_frequencies__by_model.csv \
  --label overall \
  --label gpt-4o \
  --label gpt-5 \
  --output analysis/figures/annotation_set_frequency_histogram_compare.pdf
```

## Analyses on Conversation Length

### How does conversation length affect the annotation prevalence?

A per-message regression where the outcome is the (log) number of messages remaining in the conversation after that message, and the predictors are whether that message is annotated plus a control for relative position in the conversation. The key coefficient tells us how much longer or shorter conversations are, on average, after annotated messages versus unannotated ones at the same point in the dialogue. (This is styled after a survival hazard analysis.)

```bash
python analysis/compute_annotation_post_onset_lengths.py \
      annotations/all_annotations__preprocessed.parquet \
      --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
      --length-transform log \
      --cluster-by-participant \
      --output analysis/data/annotation_remaining_length_effects.csv

python analysis/plot_annotation_hazard_effects.py \
    analysis/data/annotation_remaining_length_effects.csv \
    analysis/data/annotation_frequencies.csv \
    --output analysis/figures/annotation_remaining_length_histogram.pdf

# For extremes
python analysis/plot_annotation_hazard_effects.py \
  analysis/data/annotation_remaining_length_effects.csv \
  analysis/data/annotation_frequencies.csv \
  --output analysis/figures/annotation_remaining_length_histogram_extremes.pdf \
  --max-top 8
```

[analysis/figures/annotation_remaining_length_histogram.pdf](analysis/figures/annotation_remaining_length_histogram.pdf)

[analysis/figures/annotation_remaining_length_histogram_extremes.pdf](analysis/figures/annotation_remaining_length_histogram_extremes.pdf)

## Per-participant annotation trajectories (time and sequence)

Not part of the main analysis; this did not make it into the paper.

To generate static per-participant annotation trajectories (both time-based and sequence-based) plus overall per-annotation summaries from a full job family, use:

```bash
python -m analysis.plot_annotations_by_ppts \
  annotation_outputs/101/all_annotations.jsonl \
  --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
  --participant-ordering-json analysis/participant_ordering.json \
  --output analysis/figures
```

Using a global cutoff instead of per-annotation cutoffs:

```bash
python -m analysis.plot_annotations_by_ppts \
  annotation_outputs/101/all_annotations.jsonl \
  --participant-ordering-json analysis/participant_ordering.json \
  --llm-score-cutoff 10 \
  --output analysis/figures
```

This command:

- Discovers the full job family sharing the basename of `all_annotations.jsonl` under `annotation_outputs/`.
- Applies per-annotation LLM score cutoffs from the cutoffs CSV (or a global `--score-cutoff` when no CSV is provided).
- Uses participant ordering metadata from `analysis/participant_ordering.json` to decide when time-based plots are valid.
- Writes:
  - Per-participant figures under `analysis/figures/participants/<ppt_id>/annotations/`:
    - Time-based rolling trajectories over 5-day windows.
    - Sequence-based rolling trajectories over 20-message windows (with dual y-axes for proportions and counts).
- Overall per-annotation summary figures under `analysis/figures/annotations_overall/`, including 95% confidence intervals over participants.

Example outputs (overall summaries):

- `analysis/figures/annotations_overall/grand-significance__overall_sequence_20msgs.pdf`
- `analysis/figures/annotations_overall/grand-significance__overall_sequence_20msgs__scorecutoff10.pdf`
- `analysis/figures/annotations_overall/assistant-facilitates-violence__overall_time_5d.pdf`
- `analysis/figures/annotations_overall/assistant-facilitates-violence__overall_time_5d__scorecutoff10.pdf`

How to run multiple participants:

```
python scripts/annotation/classify_chats.py --input transcripts_de_ided \
--participant 201 \
--participant 206 \
--participant 207 \
--participant 208 \
--participant 212 \
--participant 103 \
--participant 105 \
--participant 106 \
--annotation assistant-grand-significance-ideas \
--dry-run
```

`gpt-5.1` is the default model but you can supply any OpenAI, Anthropic, TogetherAI, etc. model as interpretable by LiteLLM (so long as you have the credentials). Do not use reasoning-style models (for example, `gpt-5`) for these classification runs.

## Per-participant stacked annotation streamgraphs

Not part of the main analysis; this did not make it into the paper.

To visualize how multiple annotations co-occur over the course of each participant's interaction and in aggregate, use stacked streamgraphs:

```bash
python -m analysis.plot_annotation_streamgraphs_by_ppts \
  annotation_outputs/101/all_annotations.jsonl \
  --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
  --participant-ordering-json analysis/participant_ordering.json \
  --output analysis/figures
```

Using a global cutoff instead of per-annotation cutoffs:

```bash
python -m analysis.plot_annotation_streamgraphs_by_ppts \
  annotation_outputs/101/all_annotations.jsonl \
  --participant-ordering-json analysis/participant_ordering.json \
  --llm-score-cutoff 10 \
  --output analysis/figures
```

This command writes:

- Per-participant stacked-prevalence PDFs under `analysis/figures/participants/<ppt_id>/annotations/`, showing how the mix of annotation types evolves over normalized message index and (when dates are available) over time.
- Overall stacked-prevalence PDFs under `analysis/figures/annotations_overall/`, summarizing how annotation combinations evolve across participants.

These plots complement the time-window trajectory charts by emphasizing relative composition and co-occurrence of annotations rather than individual code trajectories.

Example outputs (overall stacked prevalence):

- `analysis/figures/annotations_overall/overall_time_stacked_raw.pdf`
- `analysis/figures/annotations_overall/overall_time_stacked_raw__scorecutoff10.pdf`

## Conversation counts (currently unused)

Not part of the main analysis; this did not make it into the paper.

To aggregate at the conversation level for a single annotation id (for example, to see how many conversations have at least `N` positive messages for that code), use:

```bash
python scripts/annotation/annotation_conversation_counts.py \
  annotation_outputs/201/20251222-192416__input=transcripts_de_ided\&max_messages=1000\&model=gpt-5.1\&preceding_context=3\&randomize=True\&randomize_per_ppt=equal.jsonl \
  --annotation-id grand-significance \
  --score-cutoff 5 \
  --min-occurrences 2
```

This scans all sibling JSONL files with the same basename under the outputs root, computes per-conversation positive and total message counts for the requested annotation, filters to conversations with at least `--min-occurrences` positives (respecting the score cutoff), and writes a CSV under `analysis/data/annotation_conversation_counts/` whose filename encodes the annotation id, cutoff, and minimum-occurrence parameters.

## Annotation profiles

Not part of the main analysis; this did not make it into the paper.

To compute participant-level annotation profiles and clustered heatmaps from a full job family, use:

```bash
python -m analysis.compute_participant_annotation_profiles \
  annotation_outputs/101/all_annotations.jsonl \
  --onset-threshold-k 5 \
  --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
  --output analysis/data/participant_annotation_profiles.csv
```

This writes:

- Participant profiles CSV: `analysis/data/participant_annotation_profiles.csv`
- Clustered heatmap PDF: `analysis/figures/participant_profiles_heatmap_participants.pdf`
- Clustering dendrogram PDF: `analysis/figures/participant_profiles_dendrogram_participants.pdf`

Notes and options:

- You can switch to `--cluster-mode annotations` to cluster annotations instead of participants.
- The onset threshold `k = 5` is encoded in the CSV columns and should be reported in captions.

## Sequential dynamics

To compute within-conversation sequential annotation dynamics (for K = 0, 1, and 10) from a full job family, first generate the per-K CSV tables:

```bash
python -m analysis.compute_sequential_annotation_dynamics \
  annotation_outputs/101/all_annotations.jsonl \
  --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
  --output-prefix analysis/data/sequential_dynamics
```

This writes per-K X->Y matrix and top-pairs CSVs under `analysis/data/` (for example, `analysis/data/sequential_dynamics_K10_matrix.csv` and `analysis/data/sequential_dynamics_K10_top_pairs.csv`).

When you instead use a global LLM score cutoff via `--llm-score-cutoff N` (and omit `--llm-cutoffs-json`), the same files are written with an explicit cutoff suffix in the prefix. For example:

```bash
python -m analysis.compute_sequential_annotation_dynamics \
  annotations/all_annotations__preprocessed.parquet \
  --llm-score-cutoff 10 \
  --window-k 0 --window-k 1 --window-k 2 --window-k 5 --window-k 10 --window-k 100
```

will write files such as `analysis/data/sequential_dynamics__scorecutoff10_K10_matrix.csv`.

To render a combined log-enrichment heatmap PDF for the same set of K values, run:

```bash
python -m analysis.plot_sequential_annotation_dynamics \
  --output-prefix analysis/data/sequential_dynamics \
  --window-k 0 --window-k 1 --window-k 10 \
  --figure-path analysis/figures/sequential_enrichment_Ks.pdf
```

This writes a single combined log-enrichment heatmap PDF for all requested K values to `analysis/figures/sequential_enrichment_Ks.pdf`.

If you used a global score cutoff, pass the suffixed prefix instead, for example:

```bash
python -m analysis.plot_sequential_annotation_dynamics \
  --output-prefix analysis/data/sequential_dynamics__scorecutoff10 \
  --window-k 0 --window-k 1 --window-k 2 --window-k 5 --window-k 10 --window-k 100 \
  --figure-path analysis/figures/sequential_enrichment_Ks__scorecutoff10.pdf
```

which uses the `sequential_dynamics__scorecutoff10_K*_matrix.csv` files as inputs for the heatmaps.

## Optional topic modeling analyses

These topic modeling commands are optional for the main paper and are intended
for exploratory analyses.

If you already have topic-modeling artifacts under
`analysis/data/annotation_topics_artifacts/`, you can generate figures and
topic-term plots with:

```bash
python -m analysis.plot_annotation_topics
```

**Outputs (figures):**

- `analysis/figures/annotation_topic_heatmap.pdf`: annotation-by-topic enrichment heatmap.
- `analysis/figures/annotation_entropy_bar.pdf`: per-annotation topic heterogeneity bar chart.
- `analysis/figures/participant_topic_heatmap.pdf`: participant-by-topic enrichment heatmap.
- `analysis/figures/topics/topic_XXX_terms.pdf`: one topic-term bar chart PDF per topic.

For the paper, these outputs can support:

- High-level tables summarizing how narrowly or broadly each annotation is distributed over topics.
- Example topic-term bar charts to illustrate the main themes behind key annotations.
- Heatmaps that show which topics are most associated with specific annotations or participants, ideally placed in the appendix.
