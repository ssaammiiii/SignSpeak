"""
Run SignSpeak live from the glove(s).

    python -m signspeak.main --port COM7
    python -m signspeak.main --port COM7 --model lda
    python -m signspeak.main --port COM7 --left-port COM8 --model lda   # two gloves
    python -m signspeak.main --demo                 # no hardware, scripted

Sign letters one at a time, hold each one steady, then sign SPEAK to have the
sentence spoken. With --left-port, raising the left hand switches to word
mode (the word signs listed in config.WORD_LABELS); dropping it returns
to letters.
"""

import argparse

from . import config, pipeline
from .sources import fake_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port of the RIGHT glove, e.g. COM7")
    ap.add_argument("--left-port", default=None,
                    help="serial port of the LEFT glove; enables word mode")
    ap.add_argument("--results", default="results",
                    help="folder containing the trained letter models")
    ap.add_argument("--model", default="bilstm",
                    choices=["bilstm", "cnn1d", "lda", "logreg",
                             "random_forest", "svm_rbf"])
    ap.add_argument("--word-results", default="results_words",
                    help="folder with the trained word model (train_words.py)")
    ap.add_argument("--word-model", default="lda",
                    choices=["lda", "random_forest", "logreg", "bilstm"])
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"confidence gate (default {config.CONFIDENCE_THRESHOLD})")
    ap.add_argument("--hold", type=int, default=4,
                    help="agreeing predictions needed before a letter is emitted")
    ap.add_argument("--motion-limit", type=float, default=0.06,
                    help="skip windows with more flex movement than this")
    ap.add_argument("--no-llm", action="store_true",
                    help="use offline word segmentation instead of Gemini")
    ap.add_argument("--demo", action="store_true",
                    help="run the scripted fake source, no hardware needed")
    args = ap.parse_args()

    if args.demo:
        source = fake_source()
    elif args.left_port:
        if not args.port:
            raise SystemExit("--port (right glove) is required with --left-port")
        from .dual import two_glove_source
        source = two_glove_source(
            right_port=args.port,
            left_port=args.left_port,
            results_dir=args.results,
            letter_model=args.model,
            word_results_dir=args.word_results,
            word_model=args.word_model,
            threshold=args.threshold,
            hold_frames=args.hold,
            motion_limit=args.motion_limit,
        )
    else:
        if not args.port:
            raise SystemExit("--port is required (or use --demo)")
        from .glove import esp32_source
        source = esp32_source(
            port=args.port,
            results_dir=args.results,
            model=args.model,
            threshold=args.threshold,
            hold_frames=args.hold,
            motion_limit=args.motion_limit,
        )

    print("\nSign letters one at a time. Hold each sign steady.")
    if args.left_port:
        print("Raise the left hand to sign a word; drop it to go back to letters.")
    print("Sign SPEAK when the word or phrase is finished.")
    print("Ctrl+C to stop.\n")

    pipeline.run(source, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
