"""
image_service.py
----------------
Everything about images, talking to Gemini over plain HTTPS (httpx),
exactly like vector_store.py does for embeddings.

    detect_intent()   -> decide: "analyze" | "edit" | "generate" | "chat"
    prepare_image()   -> validate + shrink an uploaded image
    analyze_images()  -> image(s) + question  -> text answer   (vision)
    edit_image()      -> image + instruction  -> new image     (image model)
    generate_image()  -> text prompt          -> new image     (image model)

Optional .env / Render settings (all have defaults):
    VISION_MODEL           model used to *understand* images (default: GEMINI_MODEL)
    IMAGE_MODEL            model used to *create / edit* images
                           (default: gemini-2.5-flash-image)
    IMAGE_FALLBACK_MODELS  comma separated backup image models
    MAX_IMAGE_MB           largest accepted upload (default 8)
    IMAGE_MAX_SIDE         uploads are shrunk to this many pixels (default 2048)
"""

import base64
import io
import os
import re
import time
from typing import List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]

VISION_MODEL = os.getenv("VISION_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VISION_FALLBACKS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    if m.strip()
]

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")
IMAGE_FALLBACKS = [
    m.strip() for m in os.getenv("IMAGE_FALLBACK_MODELS", "").split(",") if m.strip()
]

MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "8"))
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "2048"))

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class ImageError(Exception):
    """A problem the user can understand (shown directly in the chat)."""


# ------------------------------------------------------------------
# 1. WHAT DOES THE USER WANT?
# ------------------------------------------------------------------

_NOUNS = (
    r"(image|images|picture|pictures|photo|photos|pic|pics|illustration|logo|"
    r"poster|wallpaper|banner|icon|sketch|painting|portrait|artwork|art|"
    r"drawing|thumbnail|avatar|cartoon|render|mockup|flyer|background)"
)

# "create an image of...", "generate a logo for...", "draw a cat"
_GENERATE_RE = re.compile(
    rf"\b(create|generate|make|draw|design|produce|paint|render|imagine|give me|"
    rf"show me|i want|i need|build)\b.{{0,40}}\b{_NOUNS}\b"
    rf"|^\s*(draw|paint|sketch|illustrate)\b"
    rf"|\b{_NOUNS}\s+(of|for|showing|with)\b.{{0,5}}",
    re.I | re.S,
)

_EDIT_RE = re.compile(
    r"\b(edit|remove|erase|delete|replace|swap|change|add|insert|put|"
    r"make (it|this|the|him|her|them)|turn (it|this)|convert|transform|"
    r"enhance|improve|sharpen|blur|brighten|darken|colou?ri[sz]e|"
    r"black and white|grayscale|cartoon|anime|sketch|restyle|stylize|"
    r"background|crop|resize|retouch|fix|clean up|upscale|"
    r"cut out|recolou?r|filter|apply)\b",
    re.I,
)

# Stronger signals, used when there is NO new upload and we only
# reuse the previous picture in the chat ("make it darker").
_EDIT_FOLLOWUP_RE = re.compile(
    r"^\s*(now |also |then |please )?(make (it|this|the)|change (it|the|this)|"
    r"remove|add|replace|turn (it|this)|edit (it|this)|"
    r"brighten|darken|blur|colou?ri[sz]e|crop|enhance|zoom)\b",
    re.I,
)

_ANALYZE_RE = re.compile(
    r"^\s*(what|who|where|when|why|how many|how much|which|is |are |does |do |can you (tell|read|see|explain|describe|identify)|"
    r"describe|explain|analy[sz]e|identify|read|extract|ocr|transcribe|"
    r"summari[sz]e|count|detect|predict|classify|recogni[sz]e|compare|"
    r"tell me|check|review|caption|translate)\b",
    re.I,
)


def detect_intent(prompt: str, has_image: bool, has_previous_image: bool = False) -> str:
    """
    Returns "analyze", "edit", "generate" or "chat".

    has_image          -> the user attached a picture to THIS message
    has_previous_image -> an earlier picture exists in this chat
    """
    text = (prompt or "").strip()

    if has_image:
        if not text:
            return "analyze"                       # picture only -> describe it
        if _ANALYZE_RE.search(text) and not _EDIT_RE.search(text):
            return "analyze"
        if _EDIT_RE.search(text):
            # "what is in this image and remove the car"? treat as edit
            return "edit" if not _ANALYZE_RE.search(text) else "analyze"
        if _GENERATE_RE.search(text):
            return "edit"                          # "make an image like this" -> use it as reference
        return "analyze"

    # no new upload
    if has_previous_image and _EDIT_FOLLOWUP_RE.search(text):
        return "edit"
    if _GENERATE_RE.search(text):
        return "generate"
    return "chat"


# ------------------------------------------------------------------
# 2. PREPARE AN UPLOADED IMAGE
# ------------------------------------------------------------------

def prepare_image(data: bytes, filename: str = "image") -> Tuple[bytes, str]:
    """
    Validate an upload and shrink it if it is huge.
    Returns (bytes, mime_type). Raises ImageError for problems the user can fix.
    """
    if not data:
        raise ImageError("The image file is empty.")
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise ImageError(f"Image is larger than {MAX_IMAGE_MB} MB.")

    try:
        from PIL import Image, ImageOps
    except ImportError:
        # Pillow missing: pass the bytes through untouched
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        mime = {"png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return data, mime

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ImageError("This file could not be opened as an image.")

    fmt = (img.format or "PNG").upper()         # read before any edit loses it
    img = ImageOps.exif_transpose(img)          # fix phone-photo rotation

    changed = max(img.size) > IMAGE_MAX_SIDE
    if changed:
        img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))

    if fmt not in ("PNG", "JPEG", "WEBP"):
        fmt, changed = "PNG", True

    if not changed:
        return data, f"image/{'jpeg' if fmt == 'JPEG' else fmt.lower()}"

    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=90)
    else:
        img.save(buf, fmt)
    return buf.getvalue(), f"image/{'jpeg' if fmt == 'JPEG' else fmt.lower()}"


# ------------------------------------------------------------------
# 3. LOW-LEVEL GEMINI CALL
# ------------------------------------------------------------------

def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ImageError("GEMINI_API_KEY is missing from the environment settings.")
    return key


def _image_part(data: bytes, mime: str) -> dict:
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}


def _generate_content(models: List[str], contents: list, generation_config: dict,
                      system_text: Optional[str] = None) -> dict:
    """POST to generateContent, trying each model in turn on 404/429/5xx."""
    key = _api_key()
    last = "unknown error"

    for model in models:
        body = {"contents": contents, "generationConfig": generation_config}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        for attempt in range(2):
            try:
                resp = httpx.post(
                    f"{_BASE}/{model}:generateContent",
                    json=body,
                    headers={"x-goog-api-key": key},
                    timeout=120,
                )
            except httpx.HTTPError as exc:
                last = f"{model}: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code == 200:
                return resp.json()

            last = f"{model}: HTTP {resp.status_code} {resp.text[:300]}"
            if resp.status_code in (500, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            break  # 400/403/404/429 -> try the next model

    raise ImageError(f"The image model could not complete the request ({last}).")


def _parse(result: dict) -> Tuple[str, List[Tuple[bytes, str]]]:
    """-> (text, [(image_bytes, mime), ...])"""
    text_parts, images = [], []

    feedback = result.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        raise ImageError(
            "This request was blocked by the safety filter "
            f"({feedback['blockReason']}). Please change the prompt and try again."
        )

    for cand in result.get("candidates", []) or []:
        for part in (cand.get("content") or {}).get("parts", []) or []:
            if part.get("text"):
                text_parts.append(part["text"])
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                mime = blob.get("mimeType") or blob.get("mime_type") or "image/png"
                images.append((base64.b64decode(blob["data"]), mime))

    return "\n".join(t.strip() for t in text_parts if t.strip()), images


# ------------------------------------------------------------------
# 4. THE THREE FEATURES
# ------------------------------------------------------------------

ANALYZE_SYSTEM = (
    "You are Pearl, a careful visual assistant. Look closely at the image(s) and "
    "answer exactly what the user asks. If the user gives no question, describe the "
    "image: main subject, text visible in it, colours, setting and anything unusual. "
    "For 'predict' or 'what will happen' questions, give a reasoned prediction and say "
    "what in the image supports it. Read any visible text accurately. Never invent "
    "details you cannot see; say when something is unclear. Be concise and well organised."
)


def analyze_images(prompt: str, images: List[Tuple[bytes, str]],
                   history: Optional[List[dict]] = None) -> str:
    """Image(s) + question -> text answer."""
    contents = []
    for turn in (history or [])[-6:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = (turn.get("content") or "")[:1500]
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})

    parts = [{"text": prompt or "Describe this image in detail."}]
    parts += [_image_part(b, m) for b, m in images]
    contents.append({"role": "user", "parts": parts})

    # Gemini needs the first turn to be "user"
    while contents and contents[0]["role"] != "user":
        contents.pop(0)

    result = _generate_content(
        [VISION_MODEL] + [m for m in VISION_FALLBACKS if m != VISION_MODEL],
        contents,
        {"temperature": 0.4},
        system_text=ANALYZE_SYSTEM,
    )
    text, _ = _parse(result)
    if not text:
        raise ImageError("The model returned no answer for this image. Please try again.")
    return text


def _run_image_model(parts: list) -> Tuple[str, List[Tuple[bytes, str]]]:
    result = _generate_content(
        [IMAGE_MODEL] + [m for m in IMAGE_FALLBACKS if m != IMAGE_MODEL],
        [{"role": "user", "parts": parts}],
        {"responseModalities": ["TEXT", "IMAGE"]},
    )
    text, images = _parse(result)
    if not images:
        # The model answered in words (usually a refusal or a question)
        raise ImageError(
            text or "The model did not return an image. Try describing it differently."
        )
    return text, images


def generate_image(prompt: str) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Text prompt -> (caption, [(image_bytes, mime)])."""
    if not (prompt or "").strip():
        raise ImageError("Please describe the image you want to create.")
    return _run_image_model([{"text": prompt.strip()}])


def edit_image(prompt: str, source: List[Tuple[bytes, str]]) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Image(s) + instruction -> (caption, [(edited_image_bytes, mime)])."""
    if not source:
        raise ImageError("There is no image to edit. Please upload one.")
    instruction = (prompt or "").strip() or "Improve this image."
    parts = [{"text": instruction}] + [_image_part(b, m) for b, m in source]
    return _run_image_model(parts)
