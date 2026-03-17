"""Centralized domain schema constants for JSON and CSV payloads.

This module defines field names, column labels, and related constants for
chat messages, subset JSON payloads, subset quality records, and agreement
metrics. Import these constants instead of repeating string literals across
scripts to avoid magic strings and keep schemas consistent.
"""

from __future__ import annotations

# Chat message schema -------------------------------------------------------

MESSAGE_ROLE_KEY = "role"
MESSAGE_CONTENT_KEY = "content"
MESSAGE_TIMESTAMP_KEY = "timestamp"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


# Subset JSON schema (subsets/*.json) --------------------------------------

SUBSET_INFO_KEY = "subset_info"
SUBSET_MESSAGES_KEY = "messages"

SUBSET_INFO_ROW = "row"
SUBSET_INFO_LABEL = "label"
SUBSET_INFO_PARTICIPANT = "participant"
SUBSET_INFO_SOURCE_REL_PATH = "source_rel_path"
SUBSET_INFO_CONVERSATION_ID = "conversation_id"
SUBSET_INFO_CONVERSATION_TITLE = "conversation_title"
SUBSET_INFO_MATCH_INDEX = "match_index"
SUBSET_INFO_RANGE_INCLUSIVE = "range_inclusive"
SUBSET_INFO_SOURCE_TOTAL_MESSAGES = "source_total_messages"
SUBSET_INFO_GENERATED_UTC = "generated_utc"
SUBSET_INFO_COMMENTS = "comments"

SUBSET_INFO_PARAPHRASED = "paraphrased"
SUBSET_INFO_PARAPHRASE_VARIANT_INDEX = "paraphrase_variant_index"
SUBSET_INFO_PARAPHRASE_TOTAL_VARIANTS = "paraphrase_total_variants"

PARAPHRASE_INFO_KEY = "paraphrase_info"
PARAPHRASE_INFO_MODEL = "model"
PARAPHRASE_INFO_TEMPERATURE = "temperature"
PARAPHRASE_INFO_MAX_TOKENS = "max_tokens"
PARAPHRASE_INFO_GENERATED_UTC = "generated_utc"


# Plan CSV and summary CSV column names ------------------------------------

PLAN_COLUMN_REL_PATH = "rel_path"
PLAN_COLUMN_CONVERSATION_ID = "conversation_id"
PLAN_COLUMN_QUOTE = "quote"
PLAN_COLUMN_LABEL = "label"
PLAN_COLUMN_PARTICIPANT = "participant"
PLAN_COLUMN_PREV_COUNT = "prev_count"
PLAN_COLUMN_AFTER_COUNT = "after_count"

PLAN_REQUIRED_COLUMNS = [
    PLAN_COLUMN_REL_PATH,
    PLAN_COLUMN_CONVERSATION_ID,
    PLAN_COLUMN_QUOTE,
    PLAN_COLUMN_LABEL,
    PLAN_COLUMN_PARTICIPANT,
    PLAN_COLUMN_PREV_COUNT,
    PLAN_COLUMN_AFTER_COUNT,
]

SUMMARY_COLUMN_COMMENTS = "comments"
SUMMARY_COLUMN_PRIOR = "prior_conversation_reliance"
SUMMARY_COLUMN_UPLOADED = "uploaded_document_reliance"
SUMMARY_COLUMN_COHESION = "cohesion"
SUMMARY_COLUMN_HARMFUL_IDS = "harmful_annotation_ids"
SUMMARY_COLUMN_EARLIEST_TURN = "earliest_harmful_turn"
SUMMARY_COLUMN_PASSES_FILTERS = "passes_quality_filters"
SUMMARY_COLUMN_LLM_NOTES = "llm_notes"
SUMMARY_COLUMN_HAS_HARMFUL = "has_harmful_annotations"


# Subset quality JSONL schema (subset_quality.jsonl) -----------------------

RECORD_FIELD_TYPE = "type"
RECORD_FIELD_SUBSET_REL_PATH = "subset_rel_path"
RECORD_FIELD_ROW = "row"
RECORD_FIELD_PARTICIPANT = "participant"
RECORD_FIELD_LABEL = "label"
RECORD_FIELD_SOURCE_REL_PATH = "source_rel_path"
RECORD_FIELD_CONVERSATION_ID = "conversation_id"
RECORD_FIELD_CONVERSATION_TITLE = "conversation_title"
RECORD_FIELD_MESSAGES_COUNT = "messages_count"
RECORD_FIELD_SCORES = "scores"
RECORD_FIELD_MODEL = "model"
RECORD_FIELD_COMMENTS = "comments"
RECORD_FIELD_LLM_NOTES = "llm_notes"

RECORD_TYPE_META = "meta"
RECORD_TYPE_SUBSET_QUALITY = "subset_quality"

SCORE_FIELD_PRIOR = "prior_conversation_reliance"
SCORE_FIELD_UPLOADED = "uploaded_document_reliance"
SCORE_FIELD_COHESION = "cohesion"

META_FIELD_GENERATED_AT = "generated_at"
META_FIELD_MODEL = "model"
META_FIELD_PARTICIPANTS = "participants"
META_FIELD_ANNOTATION_IDS = "annotation_ids"
META_FIELD_INPUT_DIR = "input_dir"
META_FIELD_ARGUMENTS = "arguments"


# Agreement metrics JSON schema --------------------------------------------

AGREEMENT_FIELD_MAJORITY_CONFUSION = "majority_confusion"
AGREEMENT_AGGREGATION_ALL = "__all__"
AGREEMENT_ENTRY_KIND = "kind"
AGREEMENT_ENTRY_ANNOTATOR = "annotator"
AGREEMENT_KIND_LLM = "llm"


# Unified subset JSON schema (eval-friendly top-level fields) -----------------

SUBSET_TITLE_KEY = "title"
SUBSET_CREATE_TIME_KEY = "create_time"
SUBSET_UPDATE_TIME_KEY = "update_time"
SUBSET_USER_ID_KEY = "user_id"
SUBSET_CONVERSATION_ID_KEY = "conversation_id"
SUBSET_FIRST_MESSAGE_INDEX_KEY = "first_message_index"
SUBSET_FILENAME_HASH_KEY = "filename_hash"
SUBSET_QUALITY_SCORES_KEY = "quality_scores"

SUBSET_INFO_SELECTION_METHOD = "selection_method"
SUBSET_INFO_WINDOW_RATE = "window_rate"
SUBSET_INFO_PERCENTILE_THRESHOLD = "percentile_threshold"
SUBSET_INFO_WINDOW_SIZE = "window_size"
SUBSET_INFO_CONV_LENGTH = "conv_length"
SUBSET_INFO_IS_SHORT_CONV = "is_short_conv"

QUALITY_SCORES_PRIOR = "prior_conversation_reliance"
QUALITY_SCORES_UPLOADED = "uploaded_document_reliance"
QUALITY_SCORES_COHESION = "cohesion"
QUALITY_SCORES_MODEL = "model"
QUALITY_SCORES_SCORED_UTC = "scored_utc"
QUALITY_SCORES_NOTES = "notes"
