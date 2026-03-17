# Parsing and De-IDing

## Contents

- [Parsing the Data into User Assistant Turns](#parsing-the-data-into-user-assistant-turns)
- [De-identifying the data](#de-identifying-the-data)
- [Viewing the data](#viewing-the-data)

Note: shared Parquet assets (for example, transcripts tables) are registered in
`src/llm_delusions_data`. Prefer `llm_delusions_data.get_path(...)` or the
convenience loaders in `llm_delusions_data.loaders` when adding new scripts so
defaults stay consistent.

## Parsing the Data into User Assistant Turns

This command does a best-effort parse of the raw transcripts into user, assistant message turn json files.

```
process_chats --parse --input transcripts/01_original --verbose --output-dir transcripts/02_parsed
```

- Multiple conversations per file: if your raw file contains several chats separated by a line of dashes, supply a separator regex: `--conv-separator '(?m)^\s*---+\s*$'`. Example: `process_chats --parse --input transcripts/01_original/public --output-dir transcripts/02_parsed/public --conv-separator '(?m)^\s*---+\s*$'`.

### Conversation Branching and Path Selection

ChatGPT exports a conversation as a tree ("mapping") rather than a single thread. For HTML export, we linearize that tree by choosing one branch: if the export specifies a current_node, we follow its parent chain to the root; otherwise, we pick the deepest leaf by parent-depth and walk its ancestors. We also omit nodes flagged "hidden from conversation" (typically system/tool/context messages). This is why the HTML can differ from the JSON: the JSON contains all branches and hidden nodes; the HTML shows one visible path.

There are a variety of reasons branches appear. For example, "Regenerate response" creates sibling assistant nodes; editing and resending an earlier turn creates a new branch from that point; and tool/function calls and internal retrieval steps can add tool/system nodes. Hidden nodes are not shown in the (user-facing) UI, but they can influence the visible reply.

By default `load_chats_for_file` uses the `global_longest` strategy for ChatGPT-style exports with a `mapping` graph: it scores each node by whether it is a visible, non-hidden `user`/`assistant` turn (excluding automation-authored messages), then uses a dynamic programming pass over the tree to select the highest-scoring root-to-leaf path as the main dialogue. The alternative `all_messages` strategy keeps every visible node (including side branches, regenerations, and reminders), which is useful for exhaustive analysis but does not reflect a single conversational thread, while the `active_longest` strategy follows the chain ending at `current_node` or the deepest leaf, which can over-focus on short automation or side branches if those happened to be active when the export was taken.

Python helpers such as `load_chats_for_file` in `src/llm_delusions_annotation/chat/chat_io.py` let you load a parsed JSON transcript into reusable in-memory chat objects for downstream analysis or custom tooling.

Minimal example:

```python
from pathlib import Path

from chat.llm_delusions_annotation.chat_io import load_chats_for_file

chats = load_chats_for_file(Path("transcripts_de_ided/105/chat.html.json"))
for chat in chats:
    print(chat.key, len(chat.messages))
```

### Plan-based Parsing (`transcripts/plan.csv`)

For tricky source files you can drive parsing via a plan CSV. This gives you per-file control over which parser to use, custom role labels, and optional conversation separators.

1. Generate a starter plan for a directory (skips images, obvious binaries, `.wav`, `.mp4`, and `.DS_Store`):

```
python scripts/parse/parse_plan.py \
  --input transcripts/01_original \
  --generate-plan transcripts/plan.csv
```

2. Edit `transcripts/plan.csv` to set a `method` per file, optionally set `role_labels`, a `conv_separator` (regex or plain string), or mark a row to `skip`.

Plan CSV columns:

- `rel_path` — path relative to the input root.
- `method` — which parser to use (see list below). Use `auto` for the default behavior.
- `role_labels` — optional custom labels for turns, pipe-separated (e.g., `You|ChatGPT`).
- `conv_separator` — optional string/regex to split multiple conversations in one file.
- `skip` — set to `yes`/`true`/`1` to ignore the file.

Supported `method` values:

- `auto` — best effort. For PDFs: highlights -> boxes -> text. For DOCX: title-by-fonts -> text.
- `pdf_highlight` — PDF only. Use alternating highlight colors to infer roles.
- `pdf_boxes` — PDF only. Segment by horizontal rule "boxes" between conversations.
- `pdf_text` — PDF only. Extract text and parse by content heuristics.
- `docx_titles` — DOCX only. Use font size/style headings to find conversation titles, then parse.
- `docx_text` — DOCX only. Extract text and parse by content heuristics (normal role alternation path).
- `chatgpt_html` — ChatGPT export HTML.
- `chatgpt_json` — ChatGPT export JSON; handled as pass-through (copied as-is).

You can also force a method directly with `process_chats` (applies to all files):

```bash
process_chats --parse \
  --input transcripts/01_original \
  --output-dir tmp_parsed_pdftext \
  --method pdf_text
```

3. Run the plan:

```bash
python scripts/parse/parse_plan.py \
  --input transcripts/01_original \
  --plan transcripts/plan.csv \
  --output-dir tmp_parsed
```

Notes:

- JSON files are copied through unchanged (pass-through) and reported as `json-pass-through`.
- You can restrict work to a few files by keeping only those rows in the plan.
- When `conv_separator` is provided, text sources are split on that boundary and each segment is parsed separately.

## De-identifying the data

This command runs the parsed json files through [Microsoft Presidio](https://microsoft.github.io/presidio/) to remove identifiers and (optionally) replace them with fake identifiers.

Example de-identified outputs:

- [chat.html.json](transcripts_de_ided/105/chat.html.json) — IRB participant `105` transcript with identifiers replaced in JSON format.
- [chat.html.html](transcripts_de_ided/105/chat.html.html) — rendered HTML view of the same anonymized `105` conversation.

```bash
process_chats \
    --anon \
    --input transcripts/02_parsed/ \
    --verbose \
    --output-dir transcripts_de_ided_tmp \
    --name-allow-list '^1[0-9]+$' '^2[0-9]+$' '^hl_[0-9]+$' '^irb_[0-9]+' \
    --name-allow-list-match regex \
    --name-entities PERSON \
    --entities PERSON EMAIL_ADDRESS PHONE_NUMBER \
    --jobs 4 \
    --operator faker \
    --name-operator faker \
    --faker-locale en_US
```

Additional anonymization sources

- Contact details listed in `transcripts/metadata.csv` (any column name containing "contact") are automatically used as identifiers to anonymize (both full phrases and individual tokens, plus emails extracted from that column).

## Duplicate checking (content-based)

To flag likely duplicate transcript JSON files (by overlapping `content` strings),
use the duplicate content scanner. This is most useful for spotting re-sent
ChatGPT exports or repeated conversations across a new drop and the existing
corpus.

Example: compare a new de-identified drop against the current corpus:

```bash
python scripts/parse/find_duplicate_contents.py \
  --root transcripts_de_ided \
  --root tmp_anon \
  --min-overlap-percent 90 \
  --min-shared-lines 100
```

Notes:

- The scan only looks at JSON files and compares `content` strings.
- The overlap metric is `shared_lines / min(total_lines_in_file_a, total_lines_in_file_b)`.
- Raise `--min-overlap-percent` and `--min-shared-lines` to reduce false positives.

## Viewing the data

This command parses the generated data into interpretable html files for viewing locally.

Example viewer-friendly output: [chat.html.html](transcripts_de_ided/105/chat.html.html) — the `105` transcript formatted for quick review.

```bash
python scripts/parse/format_chats_html.py transcripts_de_ided transcripts_de_ided
```
