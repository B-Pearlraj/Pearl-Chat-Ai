"""
voice_service.py
----------------
Microphone support for Pearl.

The "mic" button inside the chat box (st.chat_input(accept_audio=True))
records a short WAV clip. This module turns that clip into text with Gemini,
using the same GEMINI_API_KEY the rest of the app already uses.

    transcribe()  -> audio bytes -> text   (Tamil, English or mixed speech)

Environment settings (all optional):
    VOICE_MODEL      Gemini model used for transcription
                     (default: VISION_MODEL / GEMINI_MODEL -> gemini-2.5-flash)
    MAX_AUDIO_MB     largest accepted recording (default 15)

Note: browsers only allow microphone access on https:// pages or on
localhost. Render gives you https automatically, so it works there.
"""

import base64
import os
from typing import List

from image_service import (
    ImageError,
    VISION_FALLBACKS,
    VISION_MODEL,
    _generate_content,
    _parse,
)

VOICE_MODEL = os.getenv("VOICE_MODEL") or VISION_MODEL
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "15"))

TRANSCRIBE_SYSTEM = (
    "You are a speech-to-text engine. Transcribe the audio exactly as spoken. "
    "Keep the speaker's language: Tamil in Tamil script, English in English, and "
    "mixed speech as it was spoken. Fix obvious punctuation and capitalisation, "
    "but never translate, summarise or answer the speech. Return ONLY the "
    "transcript, with no quotes or commentary. If there is no clear speech, "
    "return an empty string."
)


def transcribe(data: bytes, mime: str = "audio/wav") -> str:
    """Audio bytes -> transcript. Raises ImageError with a friendly message."""
    if not data:
        raise ImageError("The recording was empty. Please try again.")
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise ImageError(f"The recording is longer than {MAX_AUDIO_MB} MB. Please keep it shorter.")

    models: List[str] = [VOICE_MODEL] + [m for m in VISION_FALLBACKS if m != VOICE_MODEL]
    contents = [{
        "role": "user",
        "parts": [
            {"text": "Transcribe this recording."},
            {"inline_data": {"mime_type": mime or "audio/wav",
                             "data": base64.b64encode(data).decode()}},
        ],
    }]

    result = _generate_content(
        models, contents, {"temperature": 0}, system_text=TRANSCRIBE_SYSTEM
    )
    text, _ = _parse(result)
    return text.strip().strip('"').strip()
