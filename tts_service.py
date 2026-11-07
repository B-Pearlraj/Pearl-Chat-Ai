"""
tts_service.py
--------------
"Read aloud" for every language.

The browser's built-in voices only cover the languages installed on the
visitor's device (most computers have no Tamil, Telugu, Malayalam ... voice),
so those replies stayed silent or were read in the wrong accent.

This module asks Gemini text-to-speech for the audio instead. The model
detects the language by itself (Tamil, Hindi, Telugu, Malayalam, Kannada,
Bengali, Arabic, Chinese, Japanese, French, Spanish ... 100+ languages) and
uses the same GEMINI_API_KEY as the rest of the app.

    prefers_server_voice(text) -> True when the browser voice is not good enough
                                  (not English), so the server voice is used
    synthesize(text)           -> (wav_bytes, was_shortened)

Environment settings (all optional):
    TTS_MODEL        comma separated, tried in order
                     (default: gemini-3.8-flash-lite-tts,gemini-3.8-flash-tts)
    TTS_VOICE        prebuilt voice name (default: Kore)
    TTS_MAX_CHARS    longest text read aloud in one go (default 2400)
    TTS_CHUNK_CHARS  size of each piece sent to the model (default 800)
"""

import base64
import io
import os
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

TTS_MODELS = [
    m.strip()
    for m in os.getenv(
        "TTS_MODEL", "gemini-3.8-flash-lite-tts,gemini-3.8-flash-tts"
    ).split(",")
    if m.strip()
]
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "2400"))
CHUNK_CHARS = int(os.getenv("TTS_CHUNK_CHARS", "800"))

TTS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_RATE = 24000          # Gemini TTS output: 24 kHz, mono, 16-bit

_http = httpx.Client(timeout=90)


class TTSError(RuntimeError):
    """Raised with a message that is safe to show to the visitor."""


# ------------------------------------------------------------------
# Which messages need the server voice?
# ------------------------------------------------------------------

# Very common English words. A text that is made of Latin letters but hardly
# contains any of them is probably Spanish, French, Tanglish ... and the
# browser's English voice would read it badly.
_EN_WORDS = {
    "the", "and", "is", "are", "was", "were", "of", "to", "in", "a", "an", "for",
    "that", "with", "you", "your", "it", "this", "on", "as", "be", "can", "i",
    "will", "or", "if", "by", "from", "at", "have", "has", "not", "what", "how",
    "we", "they", "do", "but", "my", "me", "so", "there", "use", "about",
}


def prefers_server_voice(text: str) -> bool:
    """True when the reply is not English (the browser voice would fail)."""
    letters = [c for c in (text or "") if c.isalpha()]
    if len(letters) < 3:
        return False

    # Tamil, Hindi, Arabic, Chinese, Japanese, Korean, Thai, Russian ...
    non_latin = sum(1 for c in letters if ord(c) > 0x024F)
    if non_latin / len(letters) > 0.08:
        return True

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 4:
        return False
    hits = sum(1 for w in words if w in _EN_WORDS)
    return hits / len(words) < 0.10


# ------------------------------------------------------------------
# Splitting long text
# ------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?\u0964\u0965\u3002\uff01\uff1f\u061f])\s+|\n+")


def split_chunks(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Cut text into pieces of at most `limit` characters, at sentence ends."""
    pieces: List[str] = []
    current = ""

    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue

        # a single very long sentence: cut it at spaces
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        pieces.append(current)
    return pieces


# ------------------------------------------------------------------
# Talking to Gemini
# ------------------------------------------------------------------

def _pcm_from_audio(raw: bytes) -> Tuple[bytes, int]:
    """Audio bytes from the API -> (raw 16-bit PCM, sample rate)."""
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate()
    return raw, DEFAULT_RATE


def _audio_from_response(data: dict) -> Optional[str]:
    """base64 audio from an Interactions API response (last audio block)."""
    found = None
    for step in data.get("steps") or []:
        for block in step.get("content") or []:
            if block.get("type") == "audio" and block.get("data"):
                found = block["data"]
    return found


def _request_chunk(text: str, api_key: str) -> Tuple[bytes, int]:
    last = "no model answered"

    for model in TTS_MODELS:
        body = {
            "model": model,
            "input": [{
                "type": "user_input",
                "content": [{"type": "text", "text": text}],
            }],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": TTS_VOICE}]},
        }

        for attempt in range(2):
            try:
                resp = _http.post(
                    TTS_URL, json=body, headers={"x-goog-api-key": api_key}
                )
            except httpx.HTTPError as exc:
                last = f"{model}: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code == 200:
                audio = _audio_from_response(resp.json())
                if audio:
                    return _pcm_from_audio(base64.b64decode(audio))
                last = f"{model}: the response had no audio"
                break

            last = f"{model}: HTTP {resp.status_code} {resp.text[:200]}"
            if resp.status_code in (429, 500, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            break   # 400 / 403 / 404 -> try the next model

    print(f"[tts] failed: {last}")
    raise TTSError(
        "The voice could not be created right now. Please try again in a moment."
    )


def synthesize(text: str) -> Tuple[bytes, bool]:
    """
    Text (any language) -> (WAV bytes, was_shortened).
    Raises TTSError with a friendly message.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise TTSError("GEMINI_API_KEY is missing, so the voice is unavailable.")

    text = (text or "").strip()
    if not text:
        raise TTSError("There is nothing to read aloud.")

    shortened = len(text) > TTS_MAX_CHARS
    if shortened:
        cut = text.rfind(" ", 0, TTS_MAX_CHARS)
        text = text[: cut if cut > TTS_MAX_CHARS // 2 else TTS_MAX_CHARS]

    chunks = split_chunks(text)
    if not chunks:
        raise TTSError("There is nothing to read aloud.")

    if len(chunks) == 1:
        results = [_request_chunk(chunks[0], api_key)]
    else:
        with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as pool:
            results = list(pool.map(lambda c: _request_chunk(c, api_key), chunks))

    rate = results[0][1]
    pcm = b"".join(part for part, _ in results)

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return out.getvalue(), shortened
