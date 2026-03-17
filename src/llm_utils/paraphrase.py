"""
Paraphrasing helpers for generating faithful rewrites with LiteLLM.
"""

from __future__ import annotations

from typing import List, Optional

from llm_delusions_annotations.llm_utils.client import LLMClientError, completion

# TODO: we could try to have this prompt remove identifiers as well.

PARAPHRASE_SYSTEM_PROMPT = (
    "You are a careful paraphrasing assistant for transcripts.\n"
    "\n"
    "Your single job is to rewrite text provided by the user while keeping it\n"
    "meaning equivalent:\n"
    "- Preserve the exact intent, events, and safety relevant details.\n"
    "- Do not add, remove, or soften any facts or implications.\n"
    "- Keep names, dates, labels, and domain specific terms unchanged.\n"
    "- Keep the same language as the input (for example English stays English).\n"
    "- Keep roughly the same length; do not shorten aggressively.\n"
    "- Keep the same tone and register (for example casual vs formal).\n"
    "- Only change wording and sentence structure so the text is clearly\n"
    "  different from the original.\n"
    "\n"
    "You must not summarize, censor, or expand. You must not explain what you\n"
    "changed or wrap the answer in commentary. Respond with the paraphrased\n"
    "text only.\n"
)


class ParaphraseError(RuntimeError):
    """Raised when a paraphrasing request to LiteLLM fails."""


def paraphrase_block(
    text: str,
    *,
    model: str,
    temperature: float = 0.7,
    num_variants: int = 1,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
) -> List[str]:
    """Return one or more paraphrased variants for ``text``.

    Parameters
    ----------
    text:
        Input text to paraphrase.
    model:
        LiteLLM model identifier to use for paraphrasing.
    temperature:
        Sampling temperature used when generating paraphrases. Higher values
        produce more diverse rewrites.
    num_variants:
        Number of paraphrased variants to generate for the same input text.
    max_tokens:
        Optional maximum number of tokens for each paraphrased output. If
        omitted, the model default is used.
    timeout:
        Optional request timeout in seconds for the LiteLLM API.

    Returns
    -------
    List[str]
        List of paraphrased texts. The list length is at most ``num_variants``.
    """

    if not text:
        return [text] * max(num_variants, 1)

    if num_variants <= 0:
        raise ValueError("num_variants must be a positive integer")

    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    request_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "n": int(num_variants),
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = int(max_tokens)
    if timeout is not None:
        request_kwargs["timeout"] = int(timeout)

    try:
        response = completion(**request_kwargs)
    except LLMClientError as err:
        raise ParaphraseError(f"LiteLLM paraphrase request failed: {err}") from err

    choices = getattr(response, "choices", []) or []
    if not choices:
        raise ParaphraseError("LiteLLM returned no choices for paraphrase request.")

    outputs: List[str] = []
    for choice in choices:
        message_obj = getattr(choice, "message", None)
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
        else:
            content = getattr(message_obj, "content", None)
        if not isinstance(content, str):
            raise ParaphraseError("Unexpected LiteLLM choice payload structure.")
        outputs.append(content)

    return outputs
