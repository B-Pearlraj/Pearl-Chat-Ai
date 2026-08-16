import streamlit as st
import json
import ast
import re
import html as htmllib
import uuid
from datetime import datetime
import os

# ============================================================
# AGENT (LOCAL, IN-PROCESS)
#   Instead of calling a separately-running FastAPI server
#   over HTTP, we import the deep agent straight from main.py
#   and invoke it directly in this same Python process. No
#   backend server needs to be started or kept awake.
# ============================================================

from main import agent

# ============================================================
# RECURSIVE TEXT EXTRACTION  (unchanged backend logic)
# ============================================================

def extract_clean_text(obj):
    """
    Extract clean text from Deep Agent responses.
    Removes unnecessary metadata and response wrappers.
    """

    if obj is None:
        return ""

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(obj, str):

        obj_str = obj.strip()

        # Try JSON string
        if (
            (obj_str.startswith("{") and obj_str.endswith("}"))
            or
            (obj_str.startswith("[") and obj_str.endswith("]"))
        ):

            try:
                parsed = json.loads(obj_str)
                return extract_clean_text(parsed)

            except Exception:

                try:
                    parsed = ast.literal_eval(obj_str)
                    return extract_clean_text(parsed)

                except Exception:
                    pass

        return obj

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if isinstance(obj, list):

        extracted_parts = []

        for item in obj:

            result = extract_clean_text(item)

            if result:
                extracted_parts.append(result)

        return "\n\n".join(extracted_parts)

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    if isinstance(obj, dict):

        if "text" in obj and isinstance(obj["text"], str):
            return obj["text"]

        if (
            "messages" in obj
            and isinstance(obj["messages"], list)
            and len(obj["messages"]) > 0
        ):
            last_message = obj["messages"][-1]
            return extract_clean_text(last_message)

        if "content" in obj:
            return extract_clean_text(obj["content"])

        for key in ["output", "result", "response", "data"]:
            if key in obj:
                return extract_clean_text(obj[key])

        extracted_parts = []

        ignored_keys = ["extras", "signature", "type", "id", "role", "metadata"]

        for key, value in obj.items():
            if key not in ignored_keys:
                result = extract_clean_text(value)
                if result:
                    extracted_parts.append(result)

        return "\n\n".join(extracted_parts)

    return str(obj)


# ============================================================
# MARKDOWN-LITE RENDERER
#   Converts the assistant's Markdown-style output (bold,
#   italics, inline code, fenced code blocks, bullet/numbered
#   lists, headings, paragraphs) into safe, formally-styled
#   HTML for the chat bubbles. Text is HTML-escaped first, so
#   nothing but the recognised Markdown syntax below is ever
#   turned into a tag — this stays safe against injection.
# ============================================================

def markdown_to_html(text: str) -> str:

    if not text:
        return ""

    safe = htmllib.escape(text)

    # ---- Fenced code blocks: ```code``` ----
    code_blocks = []

    def _stash_code_block(match):
        code_blocks.append(match.group(1).strip("\n"))
        return f"@@PEARLCODEBLOCK{len(code_blocks) - 1}@@"

    safe = re.sub(r"```(?:[a-zA-Z0-9]*\n)?(.*?)```", _stash_code_block, safe, flags=re.DOTALL)

    # ---- Inline code: `code` ----
    safe = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", safe)

    # ---- Bold: **text** or __text__ ----
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"__(.+?)__", r"<strong>\1</strong>", safe)

    # ---- Italic: *text* or _text_ ----
    safe = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)", r"<em>\1</em>", safe)
    safe = re.sub(r"(?<!_)_(?!_)([^_\n]+?)_(?!_)", r"<em>\1</em>", safe)

    # ---- Headings: #, ##, ### ----
    safe = re.sub(
        r"^#{1,4}\s+(.+)$",
        r'<div class="pearl-section-heading">\1</div>',
        safe,
        flags=re.MULTILINE,
    )

    # ---- Walk line by line: lists, headings, paragraphs ----
    lines = safe.split("\n")
    html_parts = []
    list_buffer = []
    list_type = None
    paragraph_buffer = []

    def flush_list():
        nonlocal list_buffer, list_type
        if list_buffer:
            tag = "ul" if list_type == "ul" else "ol"
            items = "".join(f"<li>{item}</li>" for item in list_buffer)
            html_parts.append(f"<{tag}>{items}</{tag}>")
        list_buffer = []
        list_type = None

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            html_parts.append("<p>" + "<br>".join(paragraph_buffer) + "</p>")
        paragraph_buffer = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            flush_list()
            flush_paragraph()
            continue

        if stripped.startswith('<div class="pearl-section-heading">'):
            flush_list()
            flush_paragraph()
            html_parts.append(stripped)
            continue

        if stripped.startswith("@@PEARLCODEBLOCK"):
            flush_list()
            flush_paragraph()
            html_parts.append(stripped)
            continue

        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        number_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)

        if bullet_match:
            flush_paragraph()
            if list_type != "ul":
                flush_list()
                list_type = "ul"
            list_buffer.append(bullet_match.group(1))
            continue

        if number_match:
            flush_paragraph()
            if list_type != "ol":
                flush_list()
                list_type = "ol"
            list_buffer.append(number_match.group(1))
            continue

        flush_list()
        paragraph_buffer.append(stripped)

    flush_list()
    flush_paragraph()

    result = "".join(html_parts) if html_parts else f"<p>{safe}</p>"

    # ---- Restore fenced code blocks ----
    for idx, code in enumerate(code_blocks):
        result = result.replace(
            f"@@PEARLCODEBLOCK{idx}@@", f"<pre><code>{code}</code></pre>"
        )

    return result


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pearl.Ai",
    page_icon="Logo-PTS.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM — "PEARL"
#   Pearl white surfaces, nacre (mother-of-pearl) gradient
#   accents, deep amethyst ink. Display face: Fraunces.
#   Body/UI face: Manrope.
# ============================================================

PEARL_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');


/* ============================================================
   MODERN AI COLOR SYSTEM
   ============================================================ */

:root{

  --pearl-bg: #0B1020;
  --pearl-bg-alt: #11182D;

  --pearl-surface: #151D33;
  --pearl-surface-2: #1B2540;

  --ink: #F5F7FF;
  --ink-soft: #9CA8C7;

  --nacre-1: #5B5FEF;
  --nacre-2: #22D3EE;
  --nacre-3: #8B5CF6;
  --nacre-4: #38BDF8;

  --accent: #6366F1;
  --accent-deep: #A5B4FC;

  --border-pearl: rgba(148,163,184,0.16);

  --shadow-pearl:
      0 8px 30px rgba(0,0,0,0.30);

}


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
.stApp{

  background:
    radial-gradient(
      100% 80% at 50% 0%,
      #182448 0%,
      #0B1020 55%,
      #070B16 100%
    ) !important;

  font-family:'Inter', sans-serif;

  color:var(--ink);

}


/* ============================================================
   STREAMLIT CHROME
   ============================================================ */

#MainMenu,
footer,
header[data-testid="stHeader"]{

  background:transparent !important;

}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"]{

  background:
    linear-gradient(
      180deg,
      #0D1427 0%,
      #0A1020 100%
    ) !important;

  border-right:
    1px solid rgba(148,163,184,0.12) !important;

}


section[data-testid="stSidebar"] .block-container{

  padding-top:1.2rem !important;

}


/* ============================================================
   SIDEBAR BRAND
   ============================================================ */

.pearl-brand{

  display:flex;
  align-items:center;
  gap:10px;

  margin-bottom:4px;

}


.pearl-orb-mini{

  width:28px;
  height:28px;

  border-radius:50%;

  flex-shrink:0;

  background:
    conic-gradient(
      from 180deg,
      #6366F1,
      #22D3EE,
      #8B5CF6,
      #38BDF8,
      #6366F1
    );

  box-shadow:

    0 0 0 3px #0D1427,

    0 0 20px rgba(99,102,241,0.45);

  animation:
    pearl-spin 8s linear infinite;

}


.pearl-brand-name{

  font-family:'Space Grotesk',sans-serif;

  font-weight:700;

  font-size:1.1rem;

  color:#F8FAFC;

}


.pearl-brand-sub{

  font-family:'Inter',sans-serif;

  font-size:0.68rem;

  color:#7180A2;

  text-transform:uppercase;

  letter-spacing:0.13em;

  margin:
    -2px 0 18px 38px;

}


/* ============================================================
   SIDEBAR HEADINGS
   ============================================================ */

.pearl-side-heading{

  font-family:'Space Grotesk',sans-serif;

  font-size:0.82rem;

  font-weight:600;

  color:#94A3B8;

  margin:
    18px 0 8px 2px;

}


/* ============================================================
   SIDEBAR FEATURES
   ============================================================ */

.pearl-feature{

  display:flex;

  gap:10px;

  align-items:flex-start;

  padding:7px 0;

  font-size:0.78rem;

  color:#8E9BB8;

  line-height:1.45;

}


.pearl-feature .dot{

  width:7px;
  height:7px;

  border-radius:50%;

  margin-top:6px;

  flex-shrink:0;

  background:
    linear-gradient(
      135deg,
      #6366F1,
      #22D3EE
    );

  box-shadow:
    0 0 10px rgba(34,211,238,0.45);

}


.pearl-feature b{

  color:#E2E8F0;

}


/* ============================================================
   STATUS CARD
   ============================================================ */

.pearl-status-card{

  background:
    linear-gradient(
      145deg,
      #151D33,
      #11182D
    );

  border:
    1px solid rgba(148,163,184,0.14);

  border-radius:12px;

  padding:12px 14px;

  margin-top:10px;

  box-shadow:
    0 8px 25px rgba(0,0,0,0.20);

  font-size:0.78rem;

}


.pearl-status-row{

  display:flex;

  justify-content:space-between;

  padding:4px 0;

  color:#8491AF;

}


.pearl-status-row b{

  color:#E2E8F0;

}


/* ============================================================
   STATUS PILLS
   ============================================================ */

.pearl-pill{

  display:inline-block;

  padding:2px 9px;

  border-radius:999px;

  font-size:0.66rem;

  font-weight:700;

}


.pearl-pill.on{

  background:
    rgba(34,197,94,0.12);

  color:#4ADE80;

  border:
    1px solid rgba(74,222,128,0.20);

}


.pearl-pill.off{

  background:
    rgba(239,68,68,0.12);

  color:#F87171;

  border:
    1px solid rgba(248,113,113,0.20);

}


/* ============================================================
   SIDEBAR INPUT
   ============================================================ */

section[data-testid="stSidebar"]
input[type="text"]{

  background:#11182D !important;

  color:#E2E8F0 !important;

  border:
    1px solid rgba(148,163,184,0.16) !important;

  border-radius:9px !important;

  font-family:'Inter',sans-serif !important;

}


section[data-testid="stSidebar"]
input[type="text"]:focus{

  border-color:
    #6366F1 !important;

  box-shadow:
    0 0 0 1px #6366F1 !important;

}


/* ============================================================
   HISTORY
   ============================================================ */

.pearl-history-wrap{

  margin-top:4px;

}


.pearl-history-empty{

  font-size:0.76rem;

  color:#687590;

  font-style:italic;

  padding:8px 2px;

}


.pearl-history-date{

  font-size:0.64rem;

  text-transform:uppercase;

  letter-spacing:0.1em;

  color:#64718C;

  margin:
    10px 0 4px 2px;

}


.pearl-hist-row .stButton > button{

  text-align:left !important;

  justify-content:flex-start !important;

  border:
    1px solid transparent !important;

  background:
    transparent !important;

  padding:
    7px 10px !important;

  font-weight:500 !important;

  color:#AAB5CD !important;

  white-space:nowrap;

  overflow:hidden;

  text-overflow:ellipsis;

}


.pearl-hist-row .stButton > button:hover{

  background:
    rgba(99,102,241,0.10) !important;

  border-color:
    rgba(99,102,241,0.18) !important;

  color:#E2E8F0 !important;

}


.pearl-hist-row.active .stButton > button{

  background:
    linear-gradient(
      135deg,
      rgba(99,102,241,0.20),
      rgba(34,211,238,0.08)
    ) !important;

  border:
    1px solid rgba(99,102,241,0.30) !important;

  color:#C7D2FE !important;

  font-weight:700 !important;

}


.pearl-hist-del .stButton > button{

  background:transparent !important;

  border:none !important;

  color:#64718C !important;

  padding:7px 4px !important;

}


.pearl-hist-del .stButton > button:hover{

  color:#F87171 !important;

  background:
    rgba(239,68,68,0.10) !important;

}


/* ============================================================
   NEW CHAT
   ============================================================ */

.pearl-new-chat .stButton > button{

  background:
    linear-gradient(
      135deg,
      #6366F1,
      #7C3AED
    ) !important;

  border:
    1px solid rgba(129,140,248,0.40) !important;

  color:#FFFFFF !important;

  font-weight:700 !important;

  box-shadow:
    0 5px 18px rgba(99,102,241,0.25);

}


.pearl-new-chat .stButton > button:hover{

  background:
    linear-gradient(
      135deg,
      #4F46E5,
      #6D28D9
    ) !important;

  box-shadow:
    0 8px 25px rgba(99,102,241,0.35);

}


/* ============================================================
   GENERAL BUTTONS
   ============================================================ */

.stButton > button{

  background:
    #151D33 !important;

  color:#C7D2FE !important;

  border:
    1px solid rgba(148,163,184,0.15) !important;

  border-radius:9px !important;

  font-weight:600 !important;

  transition:
    all 0.15s ease;

}


.stButton > button:hover{

  border-color:
    rgba(99,102,241,0.45) !important;

  background:
    #1C2744 !important;

  color:#FFFFFF !important;

}


/* ============================================================
   MAIN HEADER
   ============================================================ */

.pearl-hero{

  display:flex;

  align-items:center;

  gap:16px;

  padding:
    6px 0 2px 0;

}


.pearl-orb-hero{

  width:46px;
  height:46px;

  border-radius:50%;

  flex-shrink:0;

  background:
    conic-gradient(
      from 180deg,
      #6366F1,
      #22D3EE,
      #8B5CF6,
      #38BDF8,
      #6366F1
    );

  box-shadow:

    0 0 0 5px #11182D,

    0 0 30px rgba(99,102,241,0.40);

  animation:
    pearl-spin 10s linear infinite;

}


.pearl-hero-title{

  font-family:'Space Grotesk',sans-serif;

  font-weight:700;

  font-size:2rem;

  margin:0;

  background:
    linear-gradient(
      100deg,
      #A5B4FC,
      #22D3EE 45%,
      #C4B5FD 70%,
      #818CF8
    );

  -webkit-background-clip:text;

  background-clip:text;

  color:transparent;

}


.pearl-hero-tagline{

  font-family:'Inter',sans-serif;

  color:#8996B3;

  font-size:0.88rem;

  margin-top:3px;

}


/* ============================================================
   ANIMATIONS
   ============================================================ */

@keyframes pearl-spin{

  from{
    filter:hue-rotate(0deg);
  }

  to{
    filter:hue-rotate(360deg);
  }

}


@keyframes pearl-pulse{

  0%,100%{
    transform:scale(1);
    opacity:1;
  }

  50%{
    transform:scale(1.12);
    opacity:0.80;
  }

}


/* ============================================================
   CHAT AREA
   ============================================================ */

.block-container{

  padding-top:1.6rem;

  max-width:880px;

}


.pearl-row{

  display:flex;

  gap:12px;

  margin:18px 0;

  align-items:flex-start;

}


.pearl-row.user{

  flex-direction:row-reverse;

}


/* ============================================================
   AVATARS
   ============================================================ */

.pearl-avatar{

  width:34px;

  height:34px;

  border-radius:50%;

  flex-shrink:0;

  margin-top:2px;

}


.pearl-avatar.assistant{

  background:
    conic-gradient(
      from 180deg,
      #6366F1,
      #22D3EE,
      #8B5CF6,
      #38BDF8,
      #6366F1
    );

  box-shadow:

    0 0 0 3px #151D33,

    0 0 18px rgba(99,102,241,0.30);

}


.pearl-avatar.user{

  background:
    linear-gradient(
      135deg,
      #6366F1,
      #7C3AED
    );

  display:flex;

  align-items:center;

  justify-content:center;

  color:#FFFFFF;

  font-family:'Inter',sans-serif;

  font-weight:700;

  font-size:0.72rem;

  box-shadow:
    0 5px 15px rgba(99,102,241,0.30);

}


/* ============================================================
   CHAT BUBBLES
   ============================================================ */

.pearl-bubble{

  max-width:74%;

  padding:12px 16px;

  font-size:0.92rem;

  line-height:1.65;

  font-family:'Inter',sans-serif;

}


.pearl-bubble.assistant{

  background:
    linear-gradient(
      145deg,
      #151D33,
      #11182D
    );

  border:
    1px solid rgba(148,163,184,0.13);

  border-radius:
    4px 18px 18px 18px;

  color:#E2E8F0;

  box-shadow:
    0 7px 25px rgba(0,0,0,0.20);

}


.pearl-bubble.user{

  background:
    linear-gradient(
      135deg,
      rgba(99,102,241,0.25),
      rgba(124,58,237,0.20)
    );

  border-radius:
    18px 4px 18px 18px;

  color:#E5E7EB;

  border:
    1px solid rgba(129,140,248,0.20);

}


/* ============================================================
   MARKDOWN
   ============================================================ */

.pearl-bubble p{

  margin:
    0 0 10px 0;

}


.pearl-bubble p:last-child{

  margin-bottom:0;

}


.pearl-bubble strong{

  color:#A5B4FC;

  font-weight:700;

}


.pearl-bubble em{

  font-style:italic;

  color:#CBD5E1;

}


.pearl-bubble code{

  background:
    rgba(99,102,241,0.13);

  padding:
    2px 6px;

  border-radius:5px;

  font-family:'Courier New',monospace;

  font-size:0.84em;

  color:#67E8F9;

}


.pearl-bubble pre{

  background:#080D19;

  color:#DCE4F5;

  padding:13px 15px;

  border-radius:10px;

  overflow-x:auto;

  margin:
    8px 0 12px 0;

  border:
    1px solid rgba(148,163,184,0.12);

  box-shadow:
    0 8px 25px rgba(0,0,0,0.30);

}


.pearl-bubble pre code{

  background:transparent;

  color:inherit;

  padding:0;

  font-size:0.84em;

}


.pearl-bubble ul,
.pearl-bubble ol{

  margin:
    0 0 12px 0;

  padding-left:22px;

}


.pearl-bubble li{

  margin-bottom:5px;

}


.pearl-bubble .pearl-section-heading{

  font-family:'Space Grotesk',sans-serif;

  font-weight:600;

  font-size:1rem;

  color:#A5B4FC;

  margin:
    14px 0 6px 0;

}


.pearl-bubble .pearl-section-heading:first-child{

  margin-top:0;

}


/* ============================================================
   THINKING
   ============================================================ */

.pearl-thinking{

  display:flex;

  align-items:center;

  gap:10px;

  color:#7F8DAA;

  font-size:0.86rem;

  font-style:italic;

}


.pearl-thinking .pearl-avatar.assistant{

  animation:
    pearl-pulse 1.1s ease-in-out infinite;

}


/* ============================================================
   ERROR
   ============================================================ */

.pearl-error{

  background:
    rgba(239,68,68,0.10);

  border:
    1px solid rgba(248,113,113,0.25);

  color:#FCA5A5;

  border-radius:10px;

  padding:
    12px 16px;

  font-size:0.86rem;

}


/* ============================================================
   EMPTY STATE
   ============================================================ */

.pearl-empty{

  text-align:center;

  padding:
    60px 20px;

  color:#8491AF;

}


.pearl-empty .pearl-orb-hero{

  margin:
    0 auto 18px auto;

  width:56px;

  height:56px;

}


.pearl-empty h3{

  font-family:'Space Grotesk',sans-serif;

  color:#E2E8F0;

  font-weight:600;

  margin-bottom:6px;

}


/* ============================================================
   CHAT INPUT
   ============================================================ */

div[data-testid="stBottomBlockContainer"]{
    background:
        linear-gradient(
            180deg,
            rgba(11,16,32,0) 0%,
            rgba(11,16,32,0.92) 35%,
            #0B1020 100%
        ) !important;

    padding-top:18px !important;
    padding-bottom:16px !important;
}


/* Main chat input wrapper */

div[data-testid="stChatInput"]{

    background:transparent !important;

    border-top:none !important;

    padding:0 !important;

}


/* Input outer container */

div[data-testid="stChatInput"] > div{

    background:
        linear-gradient(
            135deg,
            #151D33,
            #11182D
        ) !important;

    border:
        1px solid rgba(148,163,184,0.22) !important;

    border-radius:18px !important;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.28),
        0 0 0 1px rgba(99,102,241,0.04) !important;

    min-height:58px !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease !important;

}


/* Focused input */

div[data-testid="stChatInput"] > div:focus-within{

    border-color:
        rgba(99,102,241,0.65) !important;

    box-shadow:
        0 8px 35px rgba(0,0,0,0.32),
        0 0 0 3px rgba(99,102,241,0.10),
        0 0 25px rgba(99,102,241,0.08) !important;

}


/* Textarea */

div[data-testid="stChatInput"] textarea{

    background:transparent !important;

    color:#E8ECF8 !important;

    border:none !important;

    outline:none !important;

    box-shadow:none !important;

    font-family:'Inter',sans-serif !important;

    font-size:0.92rem !important;

    line-height:1.5 !important;

    padding:
        15px 16px !important;

}


/* Placeholder */

div[data-testid="stChatInput"] textarea::placeholder{

    color:#7F8BA6 !important;

    opacity:1 !important;

}


/* Send button */

div[data-testid="stChatInput"] button{

    width:40px !important;

    height:40px !important;

    min-width:40px !important;

    min-height:40px !important;

    margin-right:8px !important;

    border:none !important;

    border-radius:12px !important;

    background:
        linear-gradient(
            135deg,
            #6366F1,
            #7C3AED
        ) !important;

    color:#FFFFFF !important;

    box-shadow:
        0 5px 15px rgba(99,102,241,0.30) !important;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease,
        opacity 0.15s ease !important;

}


/* Send button hover */

div[data-testid="stChatInput"] button:hover{

    background:
        linear-gradient(
            135deg,
            #818CF8,
            #8B5CF6
        ) !important;

    transform:
        translateY(-1px) !important;

    box-shadow:
        0 7px 20px rgba(99,102,241,0.42) !important;

}


/* Send button active */

div[data-testid="stChatInput"] button:active{

    transform:
        scale(0.94) !important;

}


/* Send icon */

div[data-testid="stChatInput"] button svg{

    width:19px !important;

    height:19px !important;

}


/* Remove unnecessary inner borders */

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] textarea:focus{

    border:none !important;

    outline:none !important;

}


/* Mobile */

@media (max-width:768px){

    div[data-testid="stChatInput"] > div{

        border-radius:15px !important;

        min-height:54px !important;

    }

    div[data-testid="stChatInput"] textarea{

        font-size:0.88rem !important;

        padding:
            13px 14px !important;

    }

    div[data-testid="stChatInput"] button{

        width:36px !important;

        height:36px !important;

        min-width:36px !important;

        min-height:36px !important;

        margin-right:7px !important;

        border-radius:10px !important;

    }

}
/* ============================================================
   HIDE DEPLOY BUTTON + TOOLBAR MENU (top-right)
   ============================================================ */

#MainMenu{
  visibility:hidden !important;
}

div[data-testid="stToolbarActions"]{
  display:none !important;
}

.stAppDeployButton{
  display:none !important;
}

div[data-testid="stStatusWidget"]{
  visibility:hidden !important;
}
/* ============================================================
   CREDIT / FOOTER LINE
   ============================================================ */

.pearl-credit{

  text-align:center;

  font-family:'Inter',sans-serif;

  font-size:0.72rem;

  color:#64718C;

  letter-spacing:0.03em;

  padding:14px 0 6px 0;

}


.pearl-credit b{

  color:#94A3B8;

  font-weight:600;

}
</style>
"""

st.markdown(PEARL_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
#   Each browser session keeps its own set of conversations,
#   the way a signed-in person would see their own chat
#   history down the left side — one entry per past chat.
# ============================================================

def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New chat",
        "messages": [],
        "created": datetime.now(),
    }
    st.session_state.active_id = conv_id
    return conv_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_id" not in st.session_state or st.session_state.active_id not in st.session_state.conversations:
    if st.session_state.conversations:
        # fall back to the most recently created conversation
        st.session_state.active_id = max(
            st.session_state.conversations,
            key=lambda cid: st.session_state.conversations[cid]["created"],
        )
    else:
        new_conversation()

active_conv = st.session_state.conversations[st.session_state.active_id]


# ============================================================
# HELPERS FOR RENDERING BUBBLES
# ============================================================

def render_message_html(role, content):

    avatar_html = (
        '<div class="pearl-avatar assistant"></div>'
        if role == "assistant"
        else '<div class="pearl-avatar user">You</div>'
    )
    row_class = "assistant" if role == "assistant" else "user"
    bubble_class = "assistant" if role == "assistant" else "user"

    # Assistant replies are rendered through the Markdown-lite
    # formatter so bold text, lists, and code display properly.
    # User messages are shown as plain, literal text.
    if role == "assistant":
        body_html = markdown_to_html(content)
    else:
        body_html = "<p>" + htmllib.escape(content).replace("\n", "<br>") + "</p>"

    if role == "user":
        html_block = f"""
        <div class="pearl-row {row_class}">
          <div class="pearl-bubble {bubble_class}">{body_html}</div>
          {avatar_html}
        </div>
        """
    else:
        html_block = f"""
        <div class="pearl-row {row_class}">
          {avatar_html}
          <div class="pearl-bubble {bubble_class}">{body_html}</div>
        </div>
        """
    return html_block


def render_thinking_html():
    return """
    <div class="pearl-row assistant">
      <div class="pearl-avatar assistant"></div>
      <div class="pearl-thinking">Preparing a considered response&hellip;</div>
    </div>
    """


def render_error_html(message):
    safe = htmllib.escape(message)
    return f"""
    <div class="pearl-row assistant">
      <div class="pearl-avatar assistant"></div>
      <div class="pearl-error">{safe}</div>
    </div>
    """


# ============================================================
# SIDEBAR — CHATGPT STYLE
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # SIDEBAR HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="chatgpt-side-header">
            <div class="chatgpt-logo">
                <div class="chatgpt-logo-orb"></div>
            </div>
            <div class="chatgpt-brand">
                <div class="chatgpt-brand-title">Pearl</div>
                <div class="chatgpt-brand-subtitle">AI Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    st.markdown('<div class="chatgpt-new-chat-wrap">', unsafe_allow_html=True)

    if st.button(
        "＋  New chat",
        key="sidebar_new_chat",
        use_container_width=True,
    ):
        new_conversation()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SEARCH / HISTORY TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="chatgpt-section-title">
            <span>Recent chats</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    ordered_ids = sorted(
        st.session_state.conversations,
        key=lambda cid: st.session_state.conversations[cid]["created"],
        reverse=True,
    )

    if not ordered_ids:

        st.markdown(
            """
            <div class="chatgpt-empty-history">
                <div class="chatgpt-empty-icon">💬</div>
                <div>No conversations yet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        for cid in ordered_ids:

            conv = st.session_state.conversations[cid]

            is_active = cid == st.session_state.active_id

            title = conv["title"] or "New chat"

            # Keep sidebar title compact
            if len(title) > 32:
                title = title[:32] + "…"

            # ------------------------------------------------
            # ACTIVE CHAT
            # ------------------------------------------------

            if is_active:

                st.markdown(
                    """
                    <div class="chatgpt-history-active">
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="chatgpt-history-item">
                    """,
                    unsafe_allow_html=True,
                )

            history_col, delete_col = st.columns([6, 1])

            with history_col:

                if st.button(
                    f"✹  {title}",
                    key=f"open_chat_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.active_id = cid
                    st.rerun()

            with delete_col:

                if st.button(
                    "⋯",
                    key=f"menu_chat_{cid}",
                    use_container_width=True,
                ):
                    st.session_state[f"delete_confirm_{cid}"] = True
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            # ------------------------------------------------
            # DELETE CONFIRMATION
            # ------------------------------------------------

            if st.session_state.get(f"delete_confirm_{cid}", False):

                st.markdown(
                    """
                    <div class="chatgpt-delete-box">
                        Delete this conversation?
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:

                    if st.button(
                        "Delete",
                        key=f"confirm_delete_{cid}",
                        use_container_width=True,
                    ):

                        del st.session_state.conversations[cid]

                        st.session_state.pop(
                            f"delete_confirm_{cid}",
                            None,
                        )

                        if not st.session_state.conversations:

                            new_conversation()

                        elif cid == st.session_state.active_id:

                            st.session_state.active_id = max(
                                st.session_state.conversations,
                                key=lambda c:
                                st.session_state.conversations[c]["created"],
                            )

                        st.rerun()

                with cancel_col:

                    if st.button(
                        "Cancel",
                        key=f"cancel_delete_{cid}",
                        use_container_width=True,
                    ):

                        st.session_state.pop(
                            f"delete_confirm_{cid}",
                            None,
                        )

                        st.rerun()

    # --------------------------------------------------------
    # SIDEBAR SPACER
    # --------------------------------------------------------

    st.markdown(
        '<div class="chatgpt-sidebar-spacer"></div>',
        unsafe_allow_html=True,
    )

 
    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    with st.expander("⚡  Capabilities", expanded=False):

        st.markdown(
            """
            <div class="chatgpt-capability">
                <span>🪄</span>
                <div>
                    <b>General Consultation</b>
                    <small>Professional questions and explanations</small>
                </div>
            </div>

            <div class="chatgpt-capability">
                <span>💻</span>
                <div>
                    <b>Technical Support</b>
                    <small>Python, FastAPI, Streamlit,AWS</small>
                </div>
            </div>

            <div class="chatgpt-capability">
                <span>🌐</span>
                <div>
                    <b>Current Information</b>
                    <small>Recent information and web research</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------

import requests
import streamlit as st

with st.expander("⚙️ Settings", expanded=False):

    # Backend URL
    api_url = st.text_input(
        "Backend API URL",
        value="http://127.0.0.1:8000",
        key="backend_url"
    )

    # --------------------------------------------------------
    # CHECK BACKEND CONNECTION
    # --------------------------------------------------------

    api_connected = False

    try:
        response = requests.get(
            f"{api_url}/",
            timeout=5
        )

        if response.status_code == 200:
            api_connected = True

    except requests.exceptions.RequestException:
        api_connected = False

    # --------------------------------------------------------
    # CONNECTED
    # --------------------------------------------------------

    if api_connected:

        st.markdown(
            """
            <div class="chatgpt-status connected">
                <span class="status-dot"></span>
                <span>Backend Connected</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="pearl-backend-info connected-box">
                🟢 <b>Pearl AI is ready</b><br>
                <span>You can start asking questions.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # OFFLINE
    # --------------------------------------------------------

    else:

        st.markdown(
            """
            <div class="chatgpt-status offline">
                <span class="status-dot"></span>
                <span>Backend Offline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="pearl-backend-info offline-box">
                🟡 <b>Please wait...</b><br>
                <span>Pearl AI backend is currently unavailable.</span><br>
                <span>Start the backend and try again.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.warning("⏳ Waiting for backend connection...")
    # --------------------------------------------------------
    # CLEAR CURRENT CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑  Clear current chat",
        key="clear_current_chat",
        use_container_width=True,
    ):

        active_conv["messages"] = []

        st.rerun()
        
# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="pearl-hero">
      <div class="pearl-orb-hero"></div>
      <div>
        <p class="pearl-hero-title">Welcome to Pearl.Ai</p>
        <p class="pearl-hero-tagline">AI assistant for research, technical guidance, and professional inquiries.</p>
      </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHAT HISTORY
# ============================================================

chat_container = st.container()

with chat_container:

    if not active_conv["messages"]:
        st.markdown(
            """
            <div class="pearl-empty">
              <div class="pearl-orb-hero"></div>
              <h3>Start a new conversation</h3>
              <div>Begin below with a question, a technical issue, or a request that requires current information.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        history_html = "".join(
            render_message_html(m["role"], m["content"]) for m in active_conv["messages"]
        )
        st.markdown(history_html, unsafe_allow_html=True)


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input("ASK Anything..."):

    active_conv["messages"].append({"role": "user", "content": prompt})

    # Auto-title this chat from its first message, like a history entry
    if active_conv["title"] in ("New chat", "", None):
        short_title = prompt.strip().replace("\n", " ")
        active_conv["title"] = (short_title[:40] + "…") if len(short_title) > 40 else short_title

    with chat_container:
        st.markdown(render_message_html("user", prompt), unsafe_allow_html=True)

        thinking_placeholder = st.empty()

        thinking_placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

        try:

            # Each Streamlit conversation gets its own thread_id
            # (the conversation's UUID), so separate chats in the
            # sidebar keep separate agent memory.
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state.active_id,
                    }
                },
            )

            bot_reply = extract_clean_text(result)

            if not bot_reply:
                bot_reply = "No response could be generated for this request."

            thinking_placeholder.markdown(
                render_message_html("assistant", bot_reply), unsafe_allow_html=True
            )
            active_conv["messages"].append({"role": "assistant", "content": bot_reply})

        except Exception as e:
            error_message = f"An unexpected error occurred while running the agent: {str(e)}"
            thinking_placeholder.markdown(render_error_html(error_message), unsafe_allow_html=True)
            active_conv["messages"].append({"role": "assistant", "content": error_message})

st.markdown(
        """
        <div class="pearl-credit">
            <span class="pearl-credit-line">Crafted by <b>Muthuraj</b></span>
            <span class="pearl-credit-line">&nbsp;•&nbsp;© 2026  Pearl Chat AI &nbsp;•&nbsp;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )