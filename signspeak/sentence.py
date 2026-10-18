"""
Turn a fingerspelled letter stream into a clean, punctuated sentence.

    "iamhungrypleasegivemefood"  ->  "I am hungry, please give me food."

`build()` is the only function callers need. It uses the LLM and silently falls
back to offline segmentation when the LLM is unavailable (no key / no network).
"""

import re

from . import config

_SYSTEM_PROMPT = (
    "You are the text stage of a sign-language fingerspelling glove. You receive "
    "a lowercase stream of letters that a user spelled out one hand-shape at a "
    "time. The stream may have no spaces, or only some, and it may contain a few "
    "wrong letters from the glove misclassifying similar hand shapes. "
    "Space-separated tokens longer than one letter that are real words (for "
    "example 'doctor', 'fever', 'hospital') were signed as whole words and "
    "must be kept as they are; only the unspaced letter runs need segmenting. "
    "Segment it into real English words, repair obvious one- or two-letter "
    "errors, add correct capitalization and punctuation, and return one natural "
    "sentence. Output only the sentence — no quotes, no explanation."
)

# wordninja fuses a handful of common word pairs; undo them in the fallback.
_OFFLINE_FIXUPS = {
    "iam": "i am",
    "togo": "to go",
    "thankyou": "thank you",
    "wantto": "want to",
    "haveto": "have to",
    "goingto": "going to",
}


def _from_llm(letters: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    reply = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=letters.strip(),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=config.LLM_MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # not needed here
        ),
    )
    return reply.text.strip()


def _from_offline(letters: str) -> str:
    import wordninja

    words: list[str] = []
    for word in wordninja.split(letters.lower()):
        words.extend(_OFFLINE_FIXUPS.get(word, word).split())
    if not words:
        return ""

    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:]
    sentence = re.sub(r"\bi\b", "I", sentence)
    return sentence if sentence.endswith((".", "!", "?")) else sentence + "."


def build(letters: str, use_llm: bool = True) -> str:
    """Convert a letter stream to a sentence, preferring the LLM."""
    if use_llm:
        try:
            return _from_llm(letters)
        except Exception as error:
            print(f"[sentence] LLM unavailable ({error}); using offline fallback.")
    return _from_offline(letters)
