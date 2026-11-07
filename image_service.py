"""
image_service.py
----------------
Images for Pearl.

    Gemini    -> understands things (analysis, planning edits, writing prompts)
    Stability -> draws things (creates and edits pictures)

    detect_intent()   -> decide: "analyze" | "edit" | "generate" | "chat"
    prepare_image()   -> validate + shrink an uploaded image
    analyze_images()  -> image(s) + question -> text answer      (Gemini)
    generate_image()  -> text prompt         -> new image        (Stability)
    edit_image()      -> image + instruction -> edited image     (Stability)

Environment settings:
    STABILITY_API_KEY       REQUIRED for create / edit  (platform.stability.ai)
    STABILITY_MODEL         core (default, cheapest) | ultra | sd3.5-large |
                            sd3.5-large-turbo | sd3.5-medium
    STABILITY_EDIT_STRENGTH how strongly "restyle" edits follow the new prompt
                            instead of the original photo, 0.0-1.0 (default 0.7)
    IMAGE_PROMPT_ENHANCE    1 (default) = Gemini rewrites short prompts into rich
                            ones before drawing; 0 = send the prompt as typed
    VISION_MODEL            Gemini model for analysis (default: GEMINI_MODEL)
    MAX_IMAGE_MB            largest accepted upload (default 8)
    IMAGE_MAX_SIDE          uploads are shrunk to this many pixels (default 2048)
"""

import base64
import io
import json
import os
import re
import time
from typing import List, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv

load_dotenv()

IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]

VISION_MODEL = os.getenv("VISION_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VISION_FALLBACKS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    if m.strip()
]

STABILITY_MODEL = os.getenv("STABILITY_MODEL", "core").strip().lower()
STABILITY_EDIT_STRENGTH = min(1.0, max(0.0, float(os.getenv("STABILITY_EDIT_STRENGTH", "0.7"))))
PROMPT_ENHANCE = os.getenv("IMAGE_PROMPT_ENHANCE", "1").strip() != "0"

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


# ------------------------------------------------------------------
# 5. STABILITY AI  (creating and editing pictures)
# ------------------------------------------------------------------

_STAB = "https://api.stability.ai/v2beta/stable-image"
_ASPECTS = {"1:1", "16:9", "9:16", "3:2", "2:3", "4:5", "5:4", "21:9", "9:21"}


def _stability_error(resp) -> str:
    try:
        body = resp.json()
        detail = "; ".join(body.get("errors") or []) or body.get("message") or ""
    except Exception:
        detail = (resp.text or "")[:200]

    code = resp.status_code
    if code == 401:
        return "Stability AI rejected the API key. Check STABILITY_API_KEY in the environment settings."
    if code == 402:
        return ("Stability AI says this account has no credits left. "
                "Add credits at platform.stability.ai/account/credits and try again.")
    if code == 403:
        return ("Stability AI's safety filter blocked this request. "
                "Please change the prompt or the image and try again.")
    if code == 413:
        return "The image is too large for Stability AI. Try a smaller image."
    if code == 429:
        return "Stability AI is receiving too many requests. Wait a few seconds and try again."
    return f"Stability AI could not complete the request (HTTP {code}). {detail}".strip()


def _stability_post(path: str, fields: dict, image: Optional[Tuple[bytes, str]] = None,
                    output_format: str = "png") -> Tuple[bytes, str]:
    """POST to a Stability endpoint. Returns (image_bytes, mime)."""
    key = os.getenv("STABILITY_API_KEY")
    if not key:
        raise ImageError("STABILITY_API_KEY is missing from the environment settings.")

    data = {k: v for k, v in fields.items() if v not in (None, "")}
    data["output_format"] = output_format

    if image:
        ext = "jpg" if "jpeg" in image[1] else image[1].split("/")[-1]
        files = {"image": (f"image.{ext}", image[0], image[1])}
    else:
        files = {"none": ""}      # Stability requires multipart even without a file

    last = "no response"
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{_STAB}/{path}",
                headers={"authorization": f"Bearer {key}", "accept": "image/*"},
                files=files,
                data=data,
                timeout=120,
            )
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2)
            continue

        if resp.status_code == 200:
            if resp.headers.get("finish-reason", "").upper() == "CONTENT_FILTERED":
                raise ImageError(
                    "Stability AI's safety filter blocked the result. "
                    "Please change the prompt or the image and try again."
                )
            mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
            if not mime.startswith("image/"):
                mime = f"image/{output_format}"
            return resp.content, mime

        if resp.status_code in (500, 503) and attempt == 0:
            time.sleep(2)
            continue
        raise ImageError(_stability_error(resp))

    raise ImageError(f"Stability AI could not be reached ({last}). Please try again.")


# ------------------------------------------------------------------
# 6. GEMINI WRITES THE PROMPTS  (falls back to simple rules if Gemini fails)
# ------------------------------------------------------------------

def _json_from_gemini(system: str, parts: list) -> Optional[dict]:
    try:
        result = _generate_content(
            [VISION_MODEL] + [m for m in VISION_FALLBACKS if m != VISION_MODEL],
            [{"role": "user", "parts": parts}],
            {"temperature": 0.3, "responseMimeType": "application/json"},
            system_text=system,
        )
        text, _ = _parse(result)
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


ENHANCE_SYSTEM = (
    "You write prompts for an AI image generator. Turn the user's request into one vivid, "
    "specific English prompt (under 70 words): subject, setting, style, lighting, colours, "
    "composition. Keep every detail the user asked for. If the user wants words on the image, "
    "include them in double quotes and keep them short. Choose the best aspect ratio from "
    "1:1, 16:9, 9:16, 3:2, 2:3, 4:5, 5:4, 21:9, 9:21 (posters/phone wallpapers tall, "
    "banners/desktop wide, otherwise 1:1). Reply ONLY with JSON: "
    '{"prompt": "...", "negative_prompt": "...", "aspect_ratio": "1:1"}'
)

PLAN_SYSTEM = (
    "You plan an edit for an AI image editor. Look at the image and the user's instruction, "
    "then reply ONLY with JSON: "
    '{"operation": "remove_background" | "replace" | "restyle", '
    '"search_prompt": "...", "prompt": "...", "negative_prompt": "..."}.\n'
    "- remove_background: the user wants the background removed / a transparent cut-out.\n"
    "- replace: the user wants ONE specific thing or region changed, removed or swapped "
    "(background, sky, a car, clothes colour...). search_prompt = short phrase naming what "
    "exists now ('the red car', 'the sky', 'the background'). prompt = what should be there "
    "instead; for a removal, describe what naturally fills the space ('empty road').\n"
    "- restyle: a change to the WHOLE picture (art style, cartoon, anime, festival or "
    "greeting-card theme, mood, lighting, colours, adding decorations). prompt = a full, vivid "
    "description (under 70 words) of the final image that keeps the same subject, face, pose, "
    "clothing and composition, plus the requested changes. Put any requested text in double quotes.\n"
    "Write prompts in English. negative_prompt = things to avoid (may be empty)."
)


def _enhance_prompt(text: str) -> dict:
    plan = _json_from_gemini(ENHANCE_SYSTEM, [{"text": text}]) if PROMPT_ENHANCE else None
    prompt = (plan or {}).get("prompt") or text
    aspect = (plan or {}).get("aspect_ratio")
    return {
        "prompt": prompt,
        "negative_prompt": (plan or {}).get("negative_prompt") or "",
        "aspect_ratio": aspect if aspect in _ASPECTS else "1:1",
    }


_BG_REMOVE_RE = re.compile(
    r"(remove|delete|erase|cut out|take off|get rid of)\s+(the\s+)?(image\s+|photo\s+)?background"
    r"|transparent background|background\s+(removal|remover)",
    re.I,
)
_REPLACE_RE = re.compile(
    r"(?:replace|swap|change|turn|convert)\s+(?:the\s+)?(.+?)\s+(?:with|to|into|by)\s+(.+)$", re.I
)
_REMOVE_RE = re.compile(r"(?:remove|erase|delete|get rid of)\s+(?:the\s+)?(.+)$", re.I)
_PRONOUNS = {"it", "this", "that", "image", "picture", "photo", "everything", "him", "her", "them"}


def _plan_with_rules(instruction: str) -> dict:
    """Simple fallback when Gemini cannot plan the edit."""
    text = instruction.strip().rstrip(".!")
    if _BG_REMOVE_RE.search(text):
        return {"operation": "remove_background"}

    m = _REPLACE_RE.search(text)
    if m and m.group(1).strip().lower() not in _PRONOUNS:
        return {"operation": "replace", "search_prompt": m.group(1).strip(),
                "prompt": m.group(2).strip()}

    m = _REMOVE_RE.search(text)
    if m and m.group(1).strip().lower() not in _PRONOUNS:
        what = m.group(1).strip()
        return {"operation": "replace", "search_prompt": what,
                "prompt": f"the same scene without the {what}, natural seamless background"}

    return {"operation": "restyle", "prompt": f"{text}, high quality, detailed"}


def _plan_edit(instruction: str, source: Tuple[bytes, str]) -> dict:
    plan = _json_from_gemini(
        PLAN_SYSTEM,
        [{"text": f"User instruction: {instruction}"}, _image_part(*source)],
    )
    if plan and plan.get("operation") in ("remove_background", "replace", "restyle"):
        if plan["operation"] == "replace" and not (plan.get("search_prompt") and plan.get("prompt")):
            return _plan_with_rules(instruction)
        return plan
    return _plan_with_rules(instruction)


# ------------------------------------------------------------------
# 7. THE TWO STABILITY FEATURES
# ------------------------------------------------------------------

def generate_image(prompt: str) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Text prompt -> (caption, [(image_bytes, mime)])."""
    text = (prompt or "").strip()
    if not text:
        raise ImageError("Please describe the image you want to create.")

    plan = _enhance_prompt(text)
    fields = {
        "prompt": plan["prompt"],
        "negative_prompt": plan["negative_prompt"],
        "aspect_ratio": plan["aspect_ratio"],
    }

    if STABILITY_MODEL in ("core", "ultra"):
        path = f"generate/{STABILITY_MODEL}"
    else:                                   # sd3.5-large, sd3.5-large-turbo, sd3.5-medium ...
        path = "generate/sd3"
        fields["model"] = STABILITY_MODEL
        fields["mode"] = "text-to-image"

    data, mime = _stability_post(path, fields)
    return "", [(data, mime)]


def edit_image(prompt: str, source: List[Tuple[bytes, str]]) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Image + instruction -> (caption, [(edited_image_bytes, mime)])."""
    if not source:
        raise ImageError("There is no image to edit. Please upload one.")

    instruction = (prompt or "").strip() or "Improve this image and make it look professional"
    image = source[0]                       # Stability edits one picture at a time
    plan = _plan_edit(instruction, image)
    op = plan["operation"]
    negative = plan.get("negative_prompt") or ""

    if op == "remove_background":
        data, mime = _stability_post("edit/remove-background", {}, image=image)
        return "Here is your image with the background removed.", [(data, mime)]

    if op == "replace":
        data, mime = _stability_post(
            "edit/search-and-replace",
            {
                "prompt": plan["prompt"],
                "search_prompt": plan["search_prompt"],
                "negative_prompt": negative,
            },
            image=image,
        )
        return "", [(data, mime)]

    # restyle: keeps the layout / pose of the original, repaints it to match the prompt
    data, mime = _stability_post(
        "control/structure",
        {
            "prompt": plan.get("prompt") or instruction,
            "negative_prompt": negative,
            "control_strength": STABILITY_EDIT_STRENGTH,
        },
        image=image,
    )
    return "", [(data, mime)]
