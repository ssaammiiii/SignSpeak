"""
The SignSpeak runtime loop:

    gesture source -> gate -> token buffer -> sentence -> speech

A token is either one letter ("h") or a whole word from the two-glove
word path ("doctor", "fever"). Letters run together, words are kept
apart, and the sentence stage is told which is which.
"""

from typing import Iterable

from . import config, sentence, tts
from .sources import Gesture


def render(tokens: list[str]) -> str:
    """['r','a','j','f','e','e','l','s','sick'] -> 'rajfeels sick'"""
    out: list[str] = []
    run = ""
    for tok in tokens:
        if len(tok) == 1:
            run += tok
        else:
            if run:
                out.append(run)
                run = ""
            out.append(tok)
    if run:
        out.append(run)
    return " ".join(out).strip()


def run(source: Iterable[Gesture], use_llm: bool = True) -> None:
    """Consume gestures; speak a sentence each time 'speak' is signed."""
    buffer: list[str] = []

    for symbol, confidence in source:
        if len(symbol) == 1 and confidence < config.CONFIDENCE_THRESHOLD:
            continue

        if symbol == "neutral":
            continue
        if symbol == "speak":
            text = render(buffer)
            if text:
                tts.speak(sentence.build(text, use_llm=use_llm))
            buffer.clear()
        else:
            buffer.append(symbol)
