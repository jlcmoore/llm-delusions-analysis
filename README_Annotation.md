# Classifying (Annotating) Messages

## Contents

- [Classify Chats](#classify-chats)
- [Batch classification with provider batches](#batch-classification-with-provider-batches)
- [Summarize Positives](#summarize-positives)
- [Reviewing classification outputs](#reviewing-classification-outputs)
- [Manual message annotation](#manual-message-annotation)
- [Agreement analysis and comparison viewer](#agreement-analysis-and-comparison-viewer)
- [Exporting ML splits](#exporting-ml-splits)

Note: shared Parquet assets (for example, annotation tables) are registered in
`src/llm_delusions_data`. Prefer `llm_delusions_data.get_path(...)` or the
convenience loaders in `llm_delusions_data.loaders` when adding new scripts so
defaults stay consistent.

## LiteLLM Smoke Test

To quickly sanity-check any LiteLLM-backed model and credentials, use the
one-shot smoke test helper:

```bash
python scripts/misc/test_litellm_model.py --model vertex_ai/gemini-2.5-flash \
  --prompt "Reply with a short greeting."
```

Use `--show-raw` to print the full LiteLLM response payload and
`--no-reasoning-defaults` to disable the project's reasoning defaults.

## Classify Chats

We use the following commands to classify the deided transcripts using LLMs.

In general, you probably do not want to run this full script to annotate all of the chats with all of the annotations:

```bash
python scripts/annotation/classify_chats.py --input transcripts_de_ided
```

Notes:

- For reasoning-capable models, LiteLLM requests default to `reasoning_effort="none"` unless you override it in code.
- The classifier responses now require a `rationale` field (brief explanation) alongside `quotes` and `score`.

For example in testing you might instead just choose a subset:

```bash
python scripts/annotation/classify_chats.py --input transcripts_de_ided \
--max-messages 10000 \
--randomize \
--randomize-per-ppt equal \
--dry-run

```

To build focused datasets of strong positive examples for a set of annotations, you can ask the classifier to keep going until it finds at least `N` positive messages per selected annotation (subject to `--max-messages`) and optionally enforce a minimum score cutoff for what counts as positive:

```bash
python scripts/annotation/classify_chats.py --input transcripts_de_ided \
--max-messages 100000 \
--randomize \
--randomize-per-ppt equal \
--min-positive-per-annotation 10 \
--score-cutoff 10 \
--annotation assistant-claims-unique-understanding \
--annotation assistant-discourages-self-harm \
# ...
--annotation user-violent-intent \
--dry-run
```

- `--min-positive-per-annotation N` continues scanning until each selected annotation has at least `N` messages with one or more matches (and, when provided, `score >= --score-cutoff`), stopping earlier if `--max-messages` is reached.
- `--score-cutoff K` (0–10) optionally requires `score >= K` for a classification to count as positive for the quota logic.

When you are running `classify_chats.py` with a high score cutoff (for example `--score-cutoff 10`) and want to track how many strong positives you have per annotation across all participants sharing the same job basename, you can use the helper summary script while the job is still in progress. For example, assuming you launched a run with:

```bash
python scripts/annotation/classify_chats.py \
  --input transcripts_de_ided \
  --participant 201 \
  --annotation user-expresses-isolation \
  --model gpt-5.1 \
  --job 201_baseline_run
```

you might see an output file such as:

```text
annotation_outputs/201/20251215-103953__201_baseline_run.jsonl
```

## Batch classification with provider batches

For very large runs (for example, millions of (message, annotation) pairs) you can offload work to provider-side batch jobs using `scripts/annotation/classify_chats_batch.py`. The batch workflow has three phases:

- `submit`: select messages, build prompts, and create provider batch jobs, writing a manifest JSONL per batch. With `--dry-run`, `submit` only prepares manifests (marked as pending) and does not hit the provider.
- `enqueue`: read pending manifests and turn them into provider batch jobs without re-reading transcripts. This is useful when you want to prepare all work up front and then submit batches in multiple quota-limited runs.
- `harvest`: wait for batches to complete, then write annotation JSONL outputs that match the `classify_chats.py` format.

The batch script shares the same filtering and sampling semantics as `classify_chats.py` (participants, annotations, `--max-messages`, randomization flags), but splits the work into explicit jobs.

Minimal example (direct submit + harvest):

```bash
# 1) Submit a small batch job
python scripts/annotation/classify_chats_batch.py submit \
  --input transcripts_de_ided \
  --output annotation_outputs \
  --annotation user-endorses-delusion \
  --participant 201 \
  --max-messages 1000 \
  --batch-size 1000 \
  --job 201_user_endorses_delusion

# 2) Harvest results into annotation_outputs
python scripts/annotation/classify_chats_batch.py harvest \
  --output annotation_outputs \
  --manifest-dir annotation_outputs/batch_manifests/201_user_endorses_delusion
```

Dry-run + enqueue example (prepare everything, submit later in chunks):

```bash
# 1) Prepare all manifests without creating provider batches
python scripts/annotation/classify_chats_batch.py submit \
  --input transcripts_de_ided \
  --output annotation_outputs \
  --annotation user-endorses-delusion \
  --participant 201 \
  --max-messages 100000 \
  --batch-size 1000 \
  --job 201_user_endorses_delusion \
  --dry-run

# 2) On a given day, enqueue as many pending batches as your token budget allows
python scripts/annotation/classify_chats_batch.py enqueue \
  --manifest-dir annotation_outputs/batch_manifests/201_user_endorses_delusion \
  --max-tokens 15000000000  # 15B/day budget; use 0 to disable

# 3) Harvest completed batches as usual
python scripts/annotation/classify_chats_batch.py harvest \
  --output annotation_outputs \
  --manifest-dir annotation_outputs/batch_manifests/201_user_endorses_delusion
```

Key flags (submit/enqueue/harvest):

- `--job NAME`: required for `submit`; used to group manifests under `annotation_outputs/batch_manifests/NAME` and to name harvest outputs. Pick something short and meaningful; it should correspond roughly to the CLI arguments (participants, annotations, model).
- `--manifest-dir PATH`: required for `enqueue`/`harvest` when you want to target a specific job; points at the manifest directory created during `submit` (for example, `annotation_outputs/batch_manifests/NAME`).
- `--batch-size N`: maximum number of _tasks_ per provider batch. A task is one `(message, annotation)` pair, so the total tasks per batch is `(messages per batch) × (applicable annotations per message)`. For very large jobs, start with a modest batch size (for example, 1000) and adjust once you are comfortable with provider limits.
- `--max-messages M`: as in `classify_chats.py`, a cap on the number of messages considered. This bounds the total number of tasks as `M × (applicable annotations per message)` but does not change how those tasks are chunked into batches.
- `--participant/-p`: restricts to specific participants, as in `classify_chats.py`.
- `--annotation/-a` and `--harmful`: control which annotation configs are used; behavior matches `classify_chats.py`.
- `--dry-run` (submit only): build and validate manifests (including prompt construction, token estimation, and provider file-size checks) but mark all manifests as `pending` without creating provider batches.
- `--max-tokens` (submit/enqueue): per-run token budget enforced using `litellm.token_counter`. When the projected total tokens would exceed this value, additional batches are written or left as `pending` manifests instead of being submitted. Use `0` to disable budgeting.

### Progress and polling

- `submit` shows a progress bar (`Preparing batches`) while it walks messages and builds tasks; it logs one line per created batch and manifest.
- `harvest` polls provider batches via LiteLLM and displays a compact spinner line per manifest (for example, `Harvesting batch batch_abc... | / - \`) while waiting; Ctrl-C cleanly interrupts harvest with a friendly message and exit code `130` without dumping a traceback.

Outputs:

- Manifests are written under `annotation_outputs/batch_manifests/<job>/` as JSONL files:
  - The first line is a `type: "meta"` record describing the job, batch id, provider, endpoint, and CLI arguments.
  - Subsequent lines are `type: "task"` records describing each queued task (custom id, participant, source path, chat/message indices, annotation id/name, prompt, and preceding context).
- `harvest` writes final annotation JSONL outputs under the same `annotation_outputs` tree as `classify_chats.py`, so downstream scripts (agreement, conversation counts, viewers) can operate on them without modification.
- `harvest` never overwrites existing output files. If a target JSONL already exists, it is opened in append mode and new records are added. New `__part-XXXX` files are created only when the current shard exceeds the size cap, and the next part number is chosen by scanning the participant directory for the highest existing suffix. This means retry runs can safely reuse the same `--job` name without losing prior data.

### Additional sampling and context flags

- Use `--prefilter-conversations` to stop processing a chat if the first 10 turns have no annotation matches, which is helpful when you only expect signals later in the transcript.
- Use `--randomize-conversations` to shuffle entire conversations before sampling and keep their internal order; combine with `--max-messages` to cap request volume.
- Use `--randomize` together with `--max-messages` to sample messages; sampled messages are emitted in the same order they appear in the original JSON for readability.
- Use `--randomize-per-ppt {proportional|equal}` to control how random sampling is distributed across participants when `--randomize` is enabled:
  - `proportional` (default): allocate samples in proportion to each participant's total messages.
  - `equal`: target the same number of messages per participant (as evenly as possible), redistributing when some participants cap out.
- Use `--max-conversations N` to limit each participant to their first `N` conversations regardless of message count. This works with randomization and prefiltering.
- Use `--reverse-conversations` to walk each chat from the end toward the beginning so that prefiltering evaluates the latest turns first.
- Use `--preceding-context N` (or `-c N`) to include up to `N` earlier messages from the same conversation in the prompt as context (oldest first). The model may use this context to disambiguate the target message, but quotes must come from the target message only.
- Use `--resume-auto` to locate the latest JSONL per participant matching your current annotations and skip already-seen (message, annotation) pairs. The tool prompts for confirmation before resuming.
- The classifier pre-creates a per-participant JSONL (with a meta line) at run start so auto-resume can discover prior runs, even when a participant yields no records.
- Use `--replay-from path/to/previous.jsonl` to reuse the exact message sample from a prior run while reading messages directly from the JSONL outputs (no transcripts required); this cannot be combined with `--randomize`, `--randomize-conversations`, or `--max-conversations`. When you need to replay using the original transcript files from `--input` (for example, to restore full conversation context), add `--replay-from-source` alongside `--replay-from`.
- Use `--replay-from-manual path/to/manual.jsonl` to replay messages directly from one or more manual annotation JSONL files (no transcripts required). This re-runs LLM classification only on the annotation ids present in the manual review dataset.
- Use `--replay-all-ppts` together with `--replay-from path/to/previous.jsonl` to replay the exact same message sample across all participants for a job. The tool discovers all JSONL files under the output directory that share the same filename and annotation set as the replay file, unions their message keys, and writes new per-participant outputs using your current CLI parameters (for example, a different `--model`).

### Notes on quotes and context

- The classifier is instructed to base annotations on the target message.
- Any quoted spans returned by the model that do not appear in the target message are discarded and logged as a mismatch. This ensures that matches reflect only the target message content, regardless of provided context.
- Each classification JSONL record also includes an integer `score` field from 0 to 10 indicating how strongly the target message shows the condition (0 = clearly absent, 10 = strongly present). Downstream tools typically treat scores greater than or equal to 5 as positive for agreement analysis and viewer matching.

### Dry-run Example

Dry run example (shows prompt preview and cost estimate):

```
$ python scripts/annotation/classify_chats.py --input transcripts_de_ided \
    --participant 105 \
    --annotation assistant-offplatform-action \
    --max-messages 1 \
    --dry-run
Dry run prompt preview:

[assistant-offplatform-action] Assistant Off-Platform Action

--------------------------------------------------------------------------------
<prompt text omitted>
--------------------------------------------------------------------------------

First message payload:
[
  {
    "role": "system",
    "content": "..."
  },
  {
    "role": "user",
    "content": "..."
  }
]

Max potential cost estimate:
{
  "model": "gpt-4.1-nano",
  "max_tokens": 512,
  "model_max_tokens": 16384,
  "max_total_cost_usd": 0.0012,
  "total_request_count": 1,
  "breakdown": [
    {
      "id": "assistant-offplatform-action",

      TODO: make sure this is up to date
      "name": "Assistant Off-Platform Action",
      "max_cost_usd": 0.0012
    }
  ]
}

Metadata:
{
  "participant": "105",
  "source_path": "transcripts_de_ided/105/...",
  "applicable_annotation_ids": [
    "assistant-offplatform-action"
  ],
  "max_potential_cost_usd": 0.0012
}
```

See [annotations.py](scripts/annotations.py) for the default prompt templates referenced above.

### Batch Cost estimation and safety

- The batch script adds a second layer of protection:
  - A soft cap on provider input file size (around 90 MB) per batch, enforced during `submit` and `enqueue` via the LiteLLM batch JSONL payload.
  - A per-run token budget via `--max-tokens` that prevents you from accidentally exceeding daily quotas like the 15B tokens/day gpt-5.1 batch limit.
- For more detailed job-level token and cost summaries (including approximate prompt vs. completion token breakdown and Batch API pricing), you can:
  - Rely on the output printed at the end of `classify_chats_batch.py submit` (both real runs and `--dry-run`), which reports total estimated tokens, an approximate split into prompt vs. completion tokens, and on-demand vs. 50%-discounted batch costs using `litellm.cost_per_token` and `litellm.model_cost` for `DEFAULT_CHAT_MODEL`.

Interpreting the cost math after `submit`:

- At the end of `classify_chats_batch.py submit`, you will see a block like:

  ```text
  Token breakdown (approximate, based on MAX_CLASSIFICATION_TOKENS=256):
    Prompt tokens     (estimated): P
    Completion tokens (estimated): C

  Pricing details:
    model_cost['gpt-5.1-2025-11-13']['input_cost_per_token']  = r_in
    model_cost['gpt-5.1-2025-11-13']['output_cost_per_token'] = r_out
    ...
    total_cost      = prompt_cost + completion_cost = $X.XXXX

  Batch pricing:
    Applying a 50% Batch API discount to the on-demand total (batch_total_cost = total_cost * 0.5).
    batch_total_cost = $Y.YYYY
  ```

- The formulas are:
  - `prompt_cost     = P * r_in`
  - `completion_cost = C * r_out`
  - `total_cost      = prompt_cost + completion_cost`
  - `batch_total_cost = total_cost * 0.5` (for the 50% Batch API discount).
- For `gpt-5.1-2025-11-13`, `model_cost` gives:
  - `input_cost_per_token  = 1.25e-06` → `$1.25 / 1M` prompt tokens (on demand)
  - `output_cost_per_token = 1.00e-05` → `$10.00 / 1M` completion tokens (on demand)
  - With the 50% Batch API discount, the effective Batch prices are:
    - Prompt: `$0.625 / 1M` tokens
    - Output: `$5.00 / 1M` tokens
    - Cached input (from `cache_read_input_token_cost`) ≈ `$0.063 / 1M` tokens
- You can substitute any alternative `(P, C)` pair into these formulas (for example using an empirically measured average completion-token count per response) to explore different cost scenarios for the same model and discount factor.

### Estimating completion-token usage from prior annotation runs

- To empirically measure how many completion tokens the classifier tends to use per response, you can run:

  ```bash
  python scripts/annotation/estimate_completion_tokens.py \
    annotation_outputs/205/20251222-192416__input=transcripts_de_ided\&max_messages=1000\&model=gpt-5.1\&preceding_context=3\&randomize=True\&randomize_per_ppt=equal.jsonl
  ```

  - The script treats the provided path as a stub, discovers all matching files under `annotation_outputs/**` that share the same basename, and reconstructs the model’s JSON completion for each row (`{"score": ..., "quotes": [...]}`) from the stored `score` and `matches` fields.
  - It then uses `litellm.token_counter(model=DEFAULT_CHAT_MODEL, text=<completion_json>)` to count completion tokens per response and reports:
    - the number of files scanned and responses sampled,
    - the total completion tokens, and
    - the average completion tokens per response for `openai/gpt-5.1-2025-11-13`.
  - On a recent all-messages/all-annotations human-line sample (15 matching files for the 2025-12-22 stub), this produced:
    ```text
    Completion token statistics:
      Model: openai/gpt-5.1-2025-11-13
      Files scanned        : 15
      Responses sampled    : 16426
      Total completion tok.: 291140
      Avg tokens/response  : 17.724
    ```
    You can plug this empirical completion-token average into your budgeting and cost calculations (for example, when deciding how conservative to be relative to the `MAX_CLASSIFICATION_TOKENS` cap used in prompts).

### Summarizing annotation run stats and costs

When you have one or more completed (or partially completed) annotation runs under `annotation_outputs/`, you can summarize error statistics and approximate token usage with:

```bash
python scripts/annotation/summarize_annotation_outputs.py \
  annotation_outputs/212/three_smallest.jsonl
```

This command:

- Finds all JSONL outputs with the same basename as the reference file (for example, all `three_smallest.jsonl` files under `annotation_outputs/**`).
- Reports total result rows (requests), rows with any `error`, quote-mismatch errors, fatal errors, and summed `estimated_tokens` from the meta headers.

To also estimate dollar cost given a price per million tokens, pass `--price-per-million`, for example:

```bash
python scripts/annotation/summarize_annotation_outputs.py \
  annotation_outputs/212/three_smallest.jsonl \
  --price-per-million 2.5
```

The script assumes you are running from the repo root with the Python environment from `make init` activated so that the `annotation` and `utils` packages are importable.

## Summarize Positives

You can then summarize positives across all participants sharing that job basename with:

```bash
python scripts/annotation/summarize_annotation_positives.py \
  annotation_outputs/201/20251215-103953__201_baseline_run.jsonl \
  --score-cutoff 10
```

This discovers all sibling JSONL files with the same basename under `annotation_outputs/` (for example, other participants from the same run) and reports how many messages meet the positive threshold per annotation given the chosen score cutoff.

## Reviewing classification outputs

The interactive viewer lives in `analysis/viewer/classification_viewer.html`. We rely on the browser to fetch `src/llm_delusions_annotations/data/annotations.csv` and participant JSONL outputs, so the page must be served over HTTP (not opened directly with `file://`).

1. Start a local server from the repo root:

   ```
   make viewer
   ```

   This runs a local HTTP server (`analysis/viewer/viewer_server.py`) on http://localhost:8000 and automatically opens the viewer in your default browser. Leave the terminal running while you review.

2. Open http://localhost:8000/analysis/viewer/classification_viewer.html in your browser (if it doesn't open automatically).

3. Pick a participant from the dataset dropdown. (The viewer automatically scans `annotation_outputs/` for JSONL files produced by `classify_chats.py` and groups them by participant name, then loads `src/llm_delusions_annotations/data/annotations.csv` to populate the schema metadata. The annotation metadata is required; the page will not render without it.)

4. If no datasets appear, check that your classification outputs are saved under `annotation_outputs/<bucket>/<participant>/` as JSONL files containing a leading `{"type": "meta", ...}` line (the default layout produced by `classify_chats.py`), and refresh the page once they are present.

TODO: make sure this is up to date

5. If the page reports that `annotations.csv` failed to load, the viewer is disabled. Ensure you are serving the repository (step 1), confirm that `src/llm_delusions_annotations/data/annotations.csv` exists, and refresh once the server is reachable. Opening the HTML file directly with `file://` will never work because the browser cannot fetch the required data.

## Manual message annotation

In addition to LLM-driven classification via `classify_chats.py`, we can have human annotators apply the same annotation schema message-by-message using a browser UI that mirrors the model’s view of each message.

### Prepare manual-annotation datasets

- Use `scripts/annotation/prepare_manual_annotation_dataset.py` to derive manual datasets **directly from existing classification outputs** under `annotation_outputs/`. For example, to create a dataset aligned with all `assistant-inchat-action` classifications from `gpt-5.1`:
  ```bash
  python scripts/annotation/prepare_manual_annotation_dataset.py \
    --outputs-root annotation_outputs \
    --model gpt-5.1 \
    --annotation-id assistant-inchat-action \
    --preceding-context 3 \
    --max-items 500 \
    --randomize \
    --randomize-per-ppt equal
  ```
- Omitting `--annotation-id` includes all annotation ids present in the selected runs, and omitting `--participant` includes all participants present in those runs. When `--max-items` is provided, it caps the number of items **per annotation id** across all runs (for example, 500 items per code in the example above). Passing `--randomize --randomize-per-ppt equal` is recommended when using `--max-items`, since it randomly samples within each annotation and distributes items as evenly as possible across participants (mirroring the `classify_chats.py` sampling flow). By default, the script writes to `manual_annotation_inputs/<timestamp>__...jsonl`.
- To generate a dataset containing only messages that the LLM classified as positive (for focused false-positive review), add `--llm-positive-only` and set the cutoff with `--llm-score-cutoff` (global) or per-annotation cutoffs from agreement (recommended). To use agreement-derived per-annotation cutoffs, pass the CSV from `analysis/agreement` and keep `--llm-positive-only` enabled:

  ```bash
  python scripts/annotation/prepare_manual_annotation_dataset.py \
    --outputs-root annotation_outputs \
    --model gpt-5.1 \
    --preceding-context 3 \
    --max-items 1000 \
    --randomize \
    --randomize-per-ppt equal \
    --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
    --llm-positive-only
  ```

  Per-annotation cutoffs from `--llm-cutoffs-json` take precedence; if a specific annotation id is missing in the CSV, `--llm-score-cutoff` (when provided) is used as a fallback.

  For example, to keep only strong positives (`score >= 8`) for a single annotation:

  ```bash
  python scripts/annotation/prepare_manual_annotation_dataset.py \
    --outputs-root annotation_outputs \
    --model gpt-5.1 \
    --annotation-id user-endorses-delusion \
    --preceding-context 3 \
    --max-items 100 \
    --randomize \
    --randomize-per-ppt equal \
    --llm-positive-only \
    --llm-score-cutoff 8
  ```

  This writes a manual dataset under `manual_annotation_inputs/` where every item is an LLM-positive example for `user-endorses-delusion` at the chosen score threshold.

### How to conduct manual annotations

- **Start the annotator UI**
  - From the repo root run:
    ```bash
    make viewer
    ```
  - Open the manual annotator page in your browser:
  - http://localhost:8000/analysis/viewer/manual_annotator.html
  - The page requires `src/llm_delusions_annotations/data/annotations.csv` to be reachable over HTTP (same as the classification viewer).

- **Selecting datasets and annotator identity**
  - In the left sidebar:
    - Enter an **Annotator ID** (e.g. initials). This value:
      - Is cached in the browser (localStorage) so annotators do not need to re-enter it each time.
      - Determines where autosaved labels are written on disk (see below).
    - Use the **Manual datasets** dropdown to select a JSONL file under `manual_annotation_inputs/` generated by `prepare_manual_annotation_dataset.py`. The annotator UI treats these as "server-backed" datasets and can autosave/resume progress.
    - Alternatively, annotators can upload one or more JSONL files via the file picker; uploads are not autosaved on disk and are instead downloaded manually as a labels file.

- **Annotating messages**
  - After selecting a dataset and annotation:
    - Use the **Annotation** dropdown to pick a specific code (annotation-by-annotation pass).
    - Optionally restrict to a single participant, or keep "All participants" to walk messages in the global order encoded in the dataset.
    - The right pane shows:
      - An instruction block derived from the same scope text as `scripts/annotations.py` (without LLM-specific quoting/JSON details).
      - The current **target message** plus any preceding context messages (if `--preceding-context` was non-zero when the dataset was generated).
  - Keyboard shortcuts:
    - `J` -> mark the current message as **Yes** (matches the annotation).
    - `F` -> mark as **No**.
    - `Space` -> **Back** to the previous message.
  - The "Context" checkbox and button toggle showing/hiding preceding messages without changing the underlying dataset.

- **Where labels are saved**
  - For datasets selected from the **Manual datasets** dropdown (i.e., paths under `manual_annotation_inputs/`) with a non-empty Annotator ID:
    - Labels are **autosaved incrementally** to:
      - `manual_annotation_labels/<annotator-id>/<dataset-filename>.jsonl`
    - Each line is a JSON object keyed by a stable per-message id and includes participant, chat/message location, label (`"yes"`/`"no"`), and `labeled_at` timestamp.
    - When a label is changed (for example, after pressing Back and re-annotating), the server rewrites the label file so that each message id appears at most once (latest label wins).

- **Resuming an in-progress annotation**
  - When an annotator returns to the manual annotator UI:
    - Their **Annotator ID** is restored from localStorage.
    - After selecting the same dataset from the **Manual datasets** dropdown, the UI calls:

    - `GET /api/manual-labels?dataset_path=...&annotator_id=...`
    - If a corresponding labels file exists under `manual_annotation_labels/<annotator-id>/`, all prior labels are loaded back into the viewer.
    - The annotator is positioned at the first unlabeled message for the current annotation and participant filter, and a small badge under the Annotator ID field reports how many labels were resumed.

## Agreement analysis and comparison viewer

Once you have both human labels and LLM classifications for the same sampled messages, you can compute agreement metrics and review agreement/disagreement cases in a dedicated viewer.

### Compute agreement datasets

- Run the helper script pointing at a manual-annotation dataset plus at least two annotators (human and/or LLM):
  ```bash
  python scripts/annotation/compute_annotation_agreement.py \
    --dataset manual_annotation_inputs/20251120-215302__annotation=user-misconstrues-sentience&input=transcripts_de_ided&max-messages=500&preceding-context=3&randomize=True.jsonl \
    --manual-annotator jared \
    --llm-run annotation_outputs/201/20251120-150000__cot=True&input=transcripts_de_ided&max_messages=1000&model=gpt-4.1-nano&preceding_context=3&randomize=True&replay_all_ppts=True.jsonl \
    --llm-run annotation_outputs/201/20251120-160000__cot=True&input=transcripts_de_ided&max_messages=1000&model=gpt-5.1&preceding_context=3&randomize=True.jsonl
  ```
  When you have many LLM runs under `annotation_outputs/` and only care about a particular filename, you can point to it by basename instead of the full path. For example, to calibrate GPT‑5.1 using a union of a balanced sample and a positive-only sample, plus an additional per-annotation manual dataset:
  ```bash
  python scripts/annotation/compute_annotation_agreement.py \
    --dataset "manual_annotation_inputs/20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl" \
    --dataset "manual_annotation_inputs/20251218-091458__model=gpt-5.1&max-items=10&preceding-context=0&llm-score-cutoff=5.jsonl" \
    --manual-label-dataset "manual_annotation_inputs/20251210-083216__model=gpt-5.1&per-annotation=10.from-100.jsonl" \
    --llm-run-basename "20251222-192416__input=transcripts_de_ided&max_messages=1000&model=gpt-5.1&preceding_context=3&randomize=True&randomize_per_ppt=equal.jsonl" \
    --llm-run-basename "20251215-103953__model=gpt-5.1&preceding-context=9&positive-review.jsonl"
  ```
  In this mode, all JSONL files under `annotation_outputs/` whose filenames match the given basename(s) are treated as LLM annotators, `--dataset` values are unified into a single evaluation set (so `n` and cutoffs are computed over the combined sample), and additional manual-annotation datasets are used as side sources of human labels for any overlapping transcript locations.
- When you want to analyze a small **subset** dataset but reuse human labels from a larger manual-annotation dataset, add `--manual-label-dataset` pointing at the larger file and list the annotators you care about:
  ```bash
  python scripts/annotation/compute_annotation_agreement.py \
    --dataset manual_annotation_inputs/20251210-083216__model=gpt-5.1&per-annotation=10.from-100.jsonl \
    --manual-label-dataset manual_annotation_inputs/20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl \
    --manual-annotator jared --manual-annotator AM --manual-annotator yifan \
    --score-cutoff 8
  ```
  In this mode the primary `--dataset` defines the items being evaluated, and `--manual-label-dataset` is used only as a source of additional manual labels for those items by matching transcript locations.
- The script writes:
  - `analysis/agreement_cases__<dataset-filename>.jsonl` — one record per sampled message with per-annotator labels.
  - `analysis/agreement_metrics__<dataset-filename>.json` — agreement and confusion metrics per annotation id plus an overall summary.
    - For every pair of annotators, the JSON includes:
      - Percent agreement and Cohen’s kappa over all overlapping items.
      - Raw 2×2 counts for yes/yes, no/no, yes/no, and no/yes outcomes.
    - When at least one human annotator is present, the script also:
      - Forms a binary majority-vote label from the human annotators for each item (ties are skipped).
      - Treats this majority vote as a reference label and, for each non-human annotator (typically LLMs), reports a confusion matrix against that reference:
        - True positives, false positives, true negatives, false negatives.
        - Derived accuracy, precision, recall, and F1, both per-annotation and pooled across all annotations.
- LLM labels are binarized from the integer `score` field in the classification JSONL outputs using an integer cutoff between 0 and 10 (scores `>= cutoff` are treated as positive) so that agreement metrics are computed on a shared binary scale with human yes/no labels.
  - When one or more `--score-cutoff` values are provided explicitly, those cutoffs are evaluated as-is.
  - When no explicit `--score-cutoff` values are given, all integer cutoffs 0–10 are scanned and a single “best” cutoff is selected automatically by maximizing a metric versus the human-majority labels (default: F1).
    - You can control this selection criterion with `--optimize-cutoff-metric {f1,accuracy,kappa}`; for example, `--optimize-cutoff-metric f1` will prefer cutoffs that maximize positive-class F1 against the human-majority labels rather than overall accuracy.

### Browse agreement vs disagreement

- Start the local viewer server as usual:
  ```bash
  make viewer
  ```
- Open the agreement viewer in your browser:
- http://localhost:8000/analysis/viewer/annotation_agreement_viewer.html
- In the viewer you can:
  - Select an agreement dataset (backed by the files above).
  - Filter by annotation and participant.
  - Toggle between **agreement** cases (all annotators with labels agree) and **disagreement** cases (mixed labels).
  - Inspect per-pair agreement metrics (percent agreement, Cohen’s kappa, and raw agreement counts).
  - Review per-annotator confusion matrices and accuracy-style metrics for LLM annotators relative to the human majority vote.
  - View the full message plus preceding context alongside each annotator’s yes/no decision.

### Plot agreement precision–recall curves

- After computing agreement metrics for a dataset (which writes `analysis/agreement/<dataset-dir>/metrics.score-*.json` files), you can summarize how precision and recall versus the human majority trade off across LLM score cutoffs:
  ```bash
  python -m analysis.agreement_pr \
    --dataset "20251130-203346__model=gpt-5.1&max-items=100&preceding-context=3.jsonl"
  ```
- This script reads all available `metrics.score-*.json` files for the dataset, extracts each LLM’s precision and recall against the human-majority reference at each cutoff, and saves:
  - A precision–recall style chart with one curve per LLM annotator under:
    - `analysis/figures/agreement/agreement_pr__<dataset-dir>.png`
    - On each curve, the score cutoff that **maximizes F1 vs the human majority** (aggregated across annotations) is highlighted with a larger marker to indicate the recommended operating point.
  - Per-annotator precision–recall charts with one (faint) curve per annotation id under:
    - `analysis/figures/agreement/agreement_pr_by_annotation__<dataset-dir>__<llm-name>.png`
    - For each annotation-specific curve, the score cutoff that maximizes F1 vs the human majority for that annotation is marked with an emphasized dot, and the cutoff value is written inside the dot.

- In both cases, when choosing a single score cutoff to operate at for a given LLM (overall or per-annotation), we treat the **F1-maximizing cutoff** along the precision–recall curve as the default solution concept on the Pareto frontier.

## Exporting ML splits

After running annotations and (optionally) establishing score cutoffs via agreement analysis, you can export a binarized, ML-ready Parquet dataset containing a consistent train/test split.

The export script assigns a global `split` column (`train` or `test`, roughly 80/20 by default) to every message in the dataset. This ensures that when you train multi-task or per-annotation models, there is zero data leakage—a given message is held out consistently for every target code. Binarized labels are saved as `label__<annotation_id>` and missing codes safely remain `<NA>`.

To export a dataset using per-annotation cutoffs from a CSV file:

```bash
python scripts/data/export_ml_splits.py \
  --annotations-parquet annotations/all_annotations__preprocessed.parquet \
  --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \
  --output annotations/ml_splits.parquet
```

Alternatively, you can apply a global score cutoff (for example, `7`):

```bash
python scripts/data/export_ml_splits.py \
  --annotations-parquet annotations/all_annotations__preprocessed.parquet \
  --llm-score-cutoff 7 \
  --output annotations/ml_splits.parquet
```
