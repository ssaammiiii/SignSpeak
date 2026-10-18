"""Speak text aloud using a local, offline engine."""


def speak(text: str) -> None:
    if not text:
        return
    print(f"🔊 {text}")  # always show the transcript
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 165)  # slightly slower reads more clearly
        engine.say(text)
        engine.runAndWait()
    except Exception as error:
        print(f"[tts] audio unavailable ({error})")
