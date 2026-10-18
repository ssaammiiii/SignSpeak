"""
Gesture sources for the pipeline.

A *source* is any iterable that yields ``(symbol, confidence)`` tuples, where
``symbol`` is a letter or a control gesture ('neutral', 'space', 'speak') and
``confidence`` is a float in [0, 1]. This duck-typed contract lets the fake
source used for development be swapped for a real ESP32 reader without changing
the pipeline.
"""

from typing import Iterator, Tuple

Gesture = Tuple[str, float]


def fake_source() -> Iterator[Gesture]:
    """Replays a scripted gesture stream, one prediction at a time.

    This mirrors what the LSTM does live: it classifies ONE hand shape per
    ~1-second window and emits ONE (symbol, confidence) prediction — never a
    whole word. Each entry below is a single gesture. The letters spell
    "i am hungry"; the low-confidence entry simulates the hand moving between
    signs, so you can watch the confidence gate drop it while the word stays
    intact.
    """
    predictions: list[Gesture] = [
        ("i", 0.96),
        ("a", 0.94),
        ("m", 0.92),
        ("t", 0.41),   # hand mid-transition between signs -> gate drops this
        ("h", 0.95),
        ("u", 0.93),
        ("n", 0.97),
        ("g", 0.91),
        ("r", 0.95),
        ("y", 0.90),
        ("speak", 0.99),  # control gesture: assemble the buffer and speak it
    ]
    yield from predictions


# Real hardware source, added once the glove streams data. Same contract:
# read a sensor window, classify it, yield (symbol, confidence).
#   def esp32_source(reader, model) -> Iterator[Gesture]: ...