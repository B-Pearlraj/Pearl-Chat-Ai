"""
image_service.py
----------------
Images for Pearl.

    Gemini  -> understands images (analysis / "what is this?")
    kie.ai  -> creates and edits images (Nano Banana Pro by default)

    detect_intent()   -> decide: "analyze" | "edit" | "generate" | "chat"
    prepare_image()   -> validate + shrink an uploaded image
    analyze_images()  -> image(s) + question -> text answer      (Gemini)
    generate_image()  -> text prompt         -> new image + caption (kie.ai)
    edit_image()      -> image + instruction -> edited image     (kie.ai)

How kie.ai works (all models share it):
    1. upload the picture (edit only)      -> gets a temporary URL
    2. POST /api/v1/jobs/createTask        -> gets a taskId
    3. poll GET /api/v1/jobs/recordInfo    -> until state is success / fail
    4. download resultUrls[0]

Environment settings:
    KIE_API_KEY          REQUIRED for create / edit   (kie.ai -> API Key page)
    KIE_IMAGE_MODEL      nano-banana-pro (default, best quality) | nano-banana-2 |
                         google/nano-banana-edit (cheap, edit only)
    KIE_RESOLUTION       1K (default) | 2K | 4K    (nano-banana-pro / 2 only)
    KIE_TIMEOUT_SECONDS  how long to wait for one picture (default 240)
    VISION_MODEL         Gemini model for analysis (default: GEMINI_MODEL, then gemini-3.1-flash-lite)
    MAX_IMAGE_MB         largest accepted upload (default 8)
    IMAGE_MAX_SIDE       uploads are shrunk to this many pixels (default 2048)
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

# Same defaults as main.py, so the picture / voice features use the same
# Gemini models as the chat.
VISION_MODEL = os.getenv("VISION_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
VISION_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash"
    ).split(",")
    if m.strip()
]

KIE_MODEL = os.getenv("KIE_IMAGE_MODEL", "nano-banana-pro").strip()
KIE_RESOLUTION = os.getenv("KIE_RESOLUTION", "1K").strip().upper()
KIE_TIMEOUT = int(os.getenv("KIE_TIMEOUT_SECONDS", "240"))

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
    r"crop|resize|retouch|fix|clean up|upscale|"
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


# A question or analysis request anywhere in the sentence
# ("in this what is the background colour ?", "can you predict ...")
_QUESTION_RE = re.compile(
    r"\b(what|whats|what's|which|who|whose|where|when|why|how|predict|identify|"
    r"describe|explain|tell me|detect|count|recogni[sz]e|classify|is there|are there|"
    r"is it|does it|do you see|can you see)\b",
    re.I,
)

# "this image", "the picture", "in it" -> the user means a picture from earlier in the chat
_IMAGE_REF_RE = re.compile(
    r"\b(this|that|the|above|previous|last|earlier)\s+(image|picture|photo|pic|wallpaper|"
    r"logo|poster|illustration|drawing|painting|screenshot)\b|\bin (it|this)\b",
    re.I,
)


def _is_question(text: str) -> bool:
    return text.rstrip().endswith("?") or bool(_QUESTION_RE.search(text))


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

        wants_edit = bool(_EDIT_RE.search(text))
        starts_as_question = bool(_ANALYZE_RE.search(text))

        if wants_edit:
            # "what is in this image and remove the car"? treat as a question
            return "analyze" if starts_as_question else "edit"
        if starts_as_question or _is_question(text):
            return "analyze"                       # "in this what is the background colour ?"
        if _GENERATE_RE.search(text):
            return "edit"                          # "make an image like this" -> use it as reference
        return "analyze"

    # no new upload
    if has_previous_image and _EDIT_FOLLOWUP_RE.search(text):
        return "edit"
    if has_previous_image and _IMAGE_REF_RE.search(text) and _is_question(text) \
            and not _GENERATE_RE.search(text):
        return "analyze"                           # question about the earlier picture
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


_DISCOVERED: Optional[List[str]] = None


def _discover_models() -> List[str]:
    """Ask Google which Gemini models THIS key can call (cached once found)."""
    global _DISCOVERED
    if _DISCOVERED:
        return _DISCOVERED
    found: List[str] = []
    try:
        r = httpx.get(
            _BASE,
            params={"pageSize": 200},
            headers={"x-goog-api-key": _api_key()},
            timeout=15,
        )
        if r.status_code == 200:
            for m in r.json().get("models", []) or []:
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                name = (m.get("name") or "").replace("models/", "")
                low = name.lower()
                if "flash" in low and not any(
                    bad in low for bad in ("image", "tts", "live", "embedding", "audio", "robotics")
                ):
                    found.append(name)
    except Exception:
        pass
    found = sorted(set(found), reverse=True)[:4]      # newest versions first
    if found:
        _DISCOVERED = found
    return found


def _generate_content(models: List[str], contents: list, generation_config: dict,
                      system_text: Optional[str] = None) -> dict:
    """
    POST to generateContent, trying each model in turn on 404/429/5xx.
    If every configured model fails (for example a retired model name),
    the models this API key can really use are looked up and tried too.
    """
    key = _api_key()
    last = "unknown error"
    order = list(dict.fromkeys(models))
    looked_up = False

    while True:
        for model in order:
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

        if looked_up:
            break
        looked_up = True
        extra = [m for m in _discover_models() if m not in order]
        if not extra:
            break
        order = extra

    raise ImageError(
        "The AI model could not complete the request. Check GEMINI_MODEL in your "
        f"environment settings. (Last error: {last[:200]})"
    )


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
# 5. KIE.AI  (creating and editing pictures)
# ------------------------------------------------------------------

_KIE_API = "https://api.kie.ai/api/v1"
_KIE_UPLOAD = "https://kieai.redpandaai.co/api/file-base64-upload"
_ASPECTS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}


def _kie_headers() -> dict:
    key = os.getenv("KIE_API_KEY")
    if not key:
        raise ImageError("KIE_API_KEY is missing from the environment settings.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _kie_error(code, msg) -> str:
    """kie.ai answers HTTP 200 even on failure; the real status is body['code']."""
    msg = (msg or "").strip()
    if code == 401:
        return "kie.ai rejected the API key. Check KIE_API_KEY in the environment settings."
    if code == 402:
        return ("kie.ai says this account has no credits left. "
                "Add credits at kie.ai (Billing) and try again.")
    if code == 422:
        return f"kie.ai could not accept this request. {msg}".strip()
    if code == 429:
        return "kie.ai is receiving too many requests. Wait a few seconds and try again."
    if code == 455:
        return "kie.ai is down for maintenance right now. Please try again later."
    if code in (400, 501):
        return ("kie.ai could not create this image. It may have been blocked by the "
                f"safety filter, so try rewording the prompt. {msg}").strip()
    return f"kie.ai returned an error ({code}). {msg}".strip()


def _kie_upload(data: bytes, mime: str) -> str:
    """Upload a picture and return a temporary public URL (valid for days)."""
    ext = "jpg" if "jpeg" in mime else mime.split("/")[-1]
    body = {
        "base64Data": f"data:{mime};base64,{base64.b64encode(data).decode()}",
        "uploadPath": "pearl/uploads",
        "fileName": f"pearl-{int(time.time() * 1000)}.{ext}",
    }
    try:
        resp = requests.post(_KIE_UPLOAD, json=body, headers=_kie_headers(), timeout=90)
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise ImageError(f"Could not upload the image to kie.ai ({exc}).")

    code = payload.get("code", resp.status_code)
    url = (payload.get("data") or {}).get("downloadUrl")
    if code != 200 or not url:
        raise ImageError(_kie_error(code, payload.get("msg")))
    return url


def _kie_input(prompt: str, aspect: str, image_urls: Optional[List[str]]) -> dict:
    """Build the model-specific `input` object."""
    if "edit" in KIE_MODEL.lower():
        # google/nano-banana-edit style
        inp = {"prompt": prompt, "image_urls": image_urls or [],
               "output_format": "png", "aspect_ratio": aspect}
    else:
        # nano-banana-pro / nano-banana-2 style
        inp = {
            "prompt": prompt,
            "image_input": image_urls or [],
            "aspect_ratio": aspect,
            "resolution": KIE_RESOLUTION if KIE_RESOLUTION in ("1K", "2K", "4K") else "1K",
            "output_format": "png",
        }
    return inp


def _kie_run(prompt: str, aspect: str, image_urls: Optional[List[str]] = None
             ) -> Tuple[bytes, str]:
    """Create the task, wait for it, download the picture."""
    headers = _kie_headers()

    # 1) create the task
    try:
        resp = requests.post(
            f"{_KIE_API}/jobs/createTask",
            json={"model": KIE_MODEL, "input": _kie_input(prompt, aspect, image_urls)},
            headers=headers,
            timeout=60,
        )
        created = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise ImageError(f"kie.ai could not be reached ({exc}). Please try again.")

    if created.get("code") != 200:
        raise ImageError(_kie_error(created.get("code"), created.get("msg")))
    task_id = (created.get("data") or {}).get("taskId")
    if not task_id:
        raise ImageError("kie.ai did not return a task id. Please try again.")

    # 2) wait for the result (states: waiting, queuing, generating, success, fail)
    deadline = time.time() + KIE_TIMEOUT
    delay = 3.0
    while time.time() < deadline:
        time.sleep(delay)
        delay = min(delay + 0.5, 6.0)
        try:
            info = requests.get(
                f"{_KIE_API}/jobs/recordInfo",
                params={"taskId": task_id},
                headers=headers,
                timeout=30,
            ).json()
        except (requests.RequestException, ValueError):
            continue                      # a single failed poll is fine, try again

        if info.get("code") != 200:
            raise ImageError(_kie_error(info.get("code"), info.get("msg")))

        data = info.get("data") or {}
        state = data.get("state")

        if state == "fail":
            raise ImageError(_kie_error(
                501, data.get("failMsg") or "The model could not create this image."))

        if state == "success":
            try:
                urls = json.loads(data.get("resultJson") or "{}").get("resultUrls") or []
            except ValueError:
                urls = []
            if not urls:
                raise ImageError("kie.ai finished but returned no image. Please try again.")
            return _download(urls[0])

    raise ImageError(
        "kie.ai is taking longer than expected. Please try again in a minute "
        "(you are not charged for a task that fails)."
    )


def _download(url: str) -> Tuple[bytes, str]:
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise ImageError(f"The image was created but could not be downloaded ({exc}).")
    mime = (r.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        mime = "image/jpeg" if url.lower().split("?")[0].endswith((".jpg", ".jpeg")) else "image/png"
    return r.content, mime


# ------------------------------------------------------------------
# 6. THE TWO KIE.AI FEATURES
# ------------------------------------------------------------------

_WIDE = re.compile(r"\b(banner|desktop|wide|landscape|youtube|thumbnail|cover photo|panorama|cinematic|16:9)\b", re.I)
_TALL = re.compile(r"\b(poster|story|stories|reel|portrait|phone|mobile|flyer|9:16|vertical)\b", re.I)
# "wallpaper" is landscape (16:9, desktop) unless the user says it is for a phone
_WALLPAPER = re.compile(r"\bwall\s?p[ae]?p?e?r\b", re.I)   # also catches "wallper"
_PHONE = re.compile(r"\b(phone|mobile|iphone|android|story|stories|reel|vertical|9:16)\b", re.I)


def _pick_aspect(prompt: str) -> str:
    if _WIDE.search(prompt):
        return "16:9"
    if _WALLPAPER.search(prompt) and not _PHONE.search(prompt):
        return "16:9"
    if _TALL.search(prompt) or _WALLPAPER.search(prompt):
        return "9:16" if _PHONE.search(prompt) else "3:4"
    return "1:1"


# ------------------------------------------------------------------
# Planner: tidy the request + write the caption shown under the picture
# ------------------------------------------------------------------

PLAN_SYSTEM = (
    "You prepare requests for an image app. The user's message may contain spelling "
    "mistakes. Reply with JSON only: {{\"image_prompt\": str, \"caption\": str}}.\n"
    "image_prompt: a clear, detailed English prompt for the image model. Keep the "
    "user's intent and do not add objects they did not ask for (style, lighting and "
    "mood details are fine). If any words must appear INSIDE the picture, correct "
    "their spelling and put the exact corrected text in double quotes.\n"
    "caption: the text shown under the finished picture, in the user's language. "
    "If you corrected the spelling of words that appear inside the picture, begin "
    "with: The correct wording is **\"<corrected text>\"**. and then a blank line. "
    "Then add ONE short sentence (under 30 words) that starts with the verb "
    "{verb} and says what the picture shows and its style, e.g. "
    "\"{verb} a natural, cinematic forest wallpaper with soft morning light and "
    "rich greens.\" End with at most one emoji. No headings, no lists."
)


def plan_image(prompt: str, kind: str = "generate",
               source: Optional[List[Tuple[bytes, str]]] = None) -> Tuple[str, str]:
    """
    -> (image_prompt, caption). Never raises: if the planner fails the
    original prompt is used and a plain caption is returned.
    """
    text = (prompt or "").strip()
    fallback_caption = "Here is your edited image." if kind == "edit" else "Here is the image you asked for."
    try:
        verb = "Edited" if kind == "edit" else "Created"
        parts = [{"text": f"User request: {text}"}]
        if source:
            parts += [_image_part(b, m) for b, m in source[:1]]
        result = _generate_content(
            [VISION_MODEL] + [m for m in VISION_FALLBACKS if m != VISION_MODEL],
            [{"role": "user", "parts": parts}],
            {"temperature": 0.4, "responseMimeType": "application/json"},
            system_text=PLAN_SYSTEM.format(verb=verb),
        )
        raw, _ = _parse(result)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        data = json.loads(raw)
        image_prompt = str(data.get("image_prompt") or "").strip() or text
        caption = str(data.get("caption") or "").strip() or fallback_caption
        return image_prompt, caption
    except Exception:
        return text, fallback_caption


def generate_image(prompt: str) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Text prompt -> (caption, [(image_bytes, mime)])."""
    text = (prompt or "").strip()
    if not text:
        raise ImageError("Please describe the image you want to create.")
    image_prompt, caption = plan_image(text, "generate")
    data, mime = _kie_run(image_prompt, _pick_aspect(text))
    return caption, [(data, mime)]


def edit_image(prompt: str, source: List[Tuple[bytes, str]]) -> Tuple[str, List[Tuple[bytes, str]]]:
    """Image(s) + instruction -> (caption, [(edited_image_bytes, mime)])."""
    if not source:
        raise ImageError("There is no image to edit. Please upload one.")

    instruction = (prompt or "").strip() or "Improve this image and make it look professional."
    instruction, caption = plan_image(instruction, "edit", source)
    # Nano Banana follows plain instructions; this line keeps the person recognisable.
    full_prompt = (
        f"{instruction}\n\nKeep the same person, face, and overall composition "
        f"unless the instruction above says to change them."
    )

    urls = [_kie_upload(data, mime) for data, mime in source[:8]]
    data, mime = _kie_run(full_prompt, "auto", urls)
    return caption, [(data, mime)]
