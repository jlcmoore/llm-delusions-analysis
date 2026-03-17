# Usage Example

process_chats --input ../parser_test_transcripts/ -j 8 -v --log-file convparse.log --parse --anon

# Run GUI

python gui_app.py

# Build GUI binary

pyinstaller chatparse_gui.spec

## Common workflows

- Convert to JSON only (no anonymization)
  - `process_chats --input <INPUT_DIR> --parse --output-dir <PARSED_OUT>`
  - Result: JSON files under <PARSED_OUT> mirroring <INPUT_DIR>.
  - Force a specific parser (applies to all files), e.g., PDF via text: `process_chats --input <INPUT_DIR> --parse --output-dir <PARSED_OUT> --method pdf_text`
- Anonymize only (assumes parsed JSONs already exist)
  - `process_chats --input <PARSED_DIR> --anon --anon-output <ANON_OUT>`
  - Behavior: If files are parsed JSONs, only messages[\*].content are anonymized and written as JSON; non-JSON text files are anonymized end-to-end; binaries optionally copied.
- Do both (convert then anonymize)
  - `process_chats --input <INPUT_DIR> --parse --output-dir <PARSED_OUT> --anon --anon-output <ANON_OUT>`
  - The anonymization step uses <PARSED_OUT> automatically when both flags are present.
- Validate parsed JSONs
  - `python -m chatlog_processing_pipeline.commands validate <PARSED_DIR>`
  - Checks role alternation, string content, minimum message count.
- Single-file inputs (parse only)
  - The CLI expects a directory. Put the file in a folder and point --input at it:
    - `mkdir -p tmp_single && cp <FILE> tmp_single/`
    - `process_chats --input tmp_single --parse --output-dir tmp_single_parsed`
    - To force a method for that one file (since it’s the only input): `process_chats --input tmp_single --parse --output-dir tmp_single_parsed --method pdf_text`
  - Output: tmp_single_parsed/<FILE>.json
- Single-file inputs (anonymize only)
  - If you already have a single parsed JSON: place it in a folder and run:
    - `mkdir -p tmp_parsed && cp <FILE>.json tmp_parsed/`
    - `process_chats --input tmp_parsed --anon --anon-output tmp_parsed_anonymised`
- Strict parsing vs lenient
  - Add `--strict-parsing` to require first role=user and strict alternation; omit it for lenient.
- Split multiple conversations in one file
  - Use `--conv-separator '<REGEX>'` to split input text before parsing segments. For lines of dashes: `--conv-separator '(?m)^\s*---+\s*$'`.
- Custom role labels on one line
  - For transcripts like `Player: Hello` / `Gemini: Hi`, pass pipe-separated labels: `--role-labels 'Player:|Gemini:'`. First label maps to `user`, others to `assistant`. Works with DOCX/TXT/PDF text flows and with `--conv-separator`.
- Useful knobs for anonymization
  - Entities, thresholds, and operators: --entities, --threshold, --operator replace|redact|mask|hash, --replace-with, --mask-char, --mask-chars-to-mask, --mask-from-end
  - Names (file/dir) anonymization uses --name-\* equivalents
  - Uses contact identifiers from `transcripts/metadata.csv` automatically (columns containing "contact").
  - Text handling: --include-all, --skip-nontext, chunking: --chunk-size, --chunk-break-window
  - spaCy model length: --spacy-max-length

## Notes

- File paths: chatlog_processing_pipeline/commands.py (main flow), chatlog_processing_pipeline/redactor.py (anonymization semantics for parsed JSONs).
- Zips are supported: they’re expanded and parsed; anonymization mirrors directory structure.
