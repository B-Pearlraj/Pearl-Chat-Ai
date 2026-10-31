import streamlit as st
import json
import ast
import re
import html as htmllib
import uuid
import inspect
import time
from datetime import datetime
import os

from main import agent
import vector_store

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
    # LANGCHAIN MESSAGE OBJECTS
    #   agent.invoke(...) returns real LangChain message objects
    #   (HumanMessage, AIMessage, ...), not plain dicts. They are
    #   not instances of dict/list/str, so without this they fall
    #   through to the raw str(obj) repr at the bottom. Unwrap
    #   them via their .content attribute instead.
    # --------------------------------------------------------

    if not isinstance(obj, (str, list, dict)) and hasattr(obj, "content"):
        return extract_clean_text(obj.content)

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
/* =========================================================
   PEARL AI STATUS
   ========================================================= */

.pearl-status {
    display: flex;
    align-items: center;
    gap: 9px;

    padding: 12px 15px;
    margin-bottom: 12px;

    border-radius: 13px;

    font-size: 14px;
    font-weight: 600;
}

.pearl-status.connected {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.12);
}

.pearl-status.offline {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.12);
}


/* STATUS DOT */

.status-dot {
    width: 9px;
    height: 9px;

    min-width: 9px;

    border-radius: 50%;
}

.connected .status-dot {
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.7);
}

.offline .status-dot {
    background: #ef4444;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.7);
}


/* =========================================================
   PEARL BACKEND CARD
   ========================================================= */

.pearl-backend-info {
    display: flex;
    align-items: flex-start;

    gap: 14px;

    padding: 17px;

    border-radius: 16px;

    margin-bottom: 5px;
}

.connected-box {
    background: rgba(15, 118, 110, 0.08);
    border: 1px solid rgba(20, 184, 166, 0.25);
}

.offline-box {
    background: rgba(239, 68, 68, 0.05);
    border: 1px solid rgba(239, 68, 68, 0.20);
}


/* =========================================================
   PEARL ICON
   ========================================================= */

.pearl-icon {
    width: 38px;
    height: 38px;

    min-width: 38px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 11px;

    background: rgba(255, 255, 255, 0.08);

    font-size: 21px;
}


/* =========================================================
   TEXT
   ========================================================= */

.pearl-status-content {
    display: flex;
    flex-direction: column;

    gap: 3px;

    padding-top: 1px;
}

.pearl-status-content strong {
    font-size: 14px;
    font-weight: 700;
}

.pearl-status-content span {
    font-size: 12px;
    opacity: 0.65;
}

.pearl-status-content .ask-text {
    margin-top: 3px;
    opacity: 0.9;
    font-weight: 600;
}

.pearl-status-content .wait-text {
    margin-top: 3px;
    font-weight: 500;
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

div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]{

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

div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover{

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

div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:active{

    transform:
        scale(0.94) !important;

}


/* Send icon */

div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] svg{

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

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]{

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



/* ============================================================
   CHATGPT-STYLE "+" ATTACH BUTTON (inside the message box)
   ============================================================ */

div[data-testid="stChatInputFileUploadButton"]{
  display:flex !important;
  align-items:center !important;
  margin-left:10px !important;
}

div[data-testid="stChatInputFileUploadButton"] button{
  width:38px !important;
  height:38px !important;
  min-width:38px !important;
  min-height:38px !important;
  padding:0 !important;
  border:none !important;
  border-radius:50% !important;
  background:rgba(148,163,184,0.14) !important;
  box-shadow:none !important;
  color:#E2E8F0 !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  transition:background 0.15s ease, transform 0.15s ease !important;
}

div[data-testid="stChatInputFileUploadButton"] button:hover{
  background:rgba(99,102,241,0.30) !important;
  transform:none !important;
}

/* hide Streamlit's paperclip icon and draw a "+" instead */
div[data-testid="stChatInputFileUploadButton"] button > *{
  display:none !important;
}

div[data-testid="stChatInputFileUploadButton"] button::before{
  content:"+";
  font-family:'Inter',sans-serif;
  font-size:26px;
  font-weight:400;
  line-height:1;
  color:#E2E8F0;
  margin-top:-2px;
}

/* ============================================================
   ATTACHED-FILE CHIPS (inside chat bubbles)
   ============================================================ */

.pearl-files{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-bottom:8px;
}

.pearl-file-chip{
  display:inline-flex;
  align-items:center;
  gap:9px;
  max-width:100%;
  padding:7px 12px;
  border-radius:12px;
  background:rgba(11,16,32,0.55);
  border:1px solid rgba(148,163,184,0.20);
  font-size:0.8rem;
  color:#E2E8F0;
}

.pearl-file-chip .ico{ font-size:1.05rem; }

.pearl-file-chip .meta{
  display:flex;
  flex-direction:column;
  min-width:0;
}

.pearl-file-chip .name{
  font-weight:600;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

.pearl-file-chip .note{
  font-size:0.68rem;
  color:#9CA8C7;
}

.pearl-file-chip.ok{ border-color:rgba(74,222,128,0.35); }
.pearl-file-chip.ok .note{ color:#4ADE80; }
.pearl-file-chip.warn{ border-color:rgba(250,204,21,0.35); }
.pearl-file-chip.warn .note{ color:#FACC15; }
.pearl-file-chip.err{ border-color:rgba(248,113,113,0.40); }
.pearl-file-chip.err .note{ color:#F87171; }
.pearl-file-chip.pending .note{ color:#A5B4FC; }

/* ============================================================
   SIDEBAR (the chatgpt-* classes used by the markup)
   ============================================================ */

.chatgpt-side-header{
  display:flex;
  align-items:center;
  gap:12px;
  margin:4px 0 16px 0;
}

.chatgpt-logo-orb{
  width:34px;
  height:34px;
  border-radius:50%;
  background:conic-gradient(from 180deg,#6366F1,#22D3EE,#8B5CF6,#38BDF8,#6366F1);
  box-shadow:0 0 0 3px #0D1427, 0 0 20px rgba(99,102,241,0.45);
  animation:pearl-spin 8s linear infinite;
}

.chatgpt-brand-title{
  font-family:'Space Grotesk',sans-serif;
  font-weight:700;
  font-size:1.15rem;
  color:#F8FAFC;
  line-height:1.1;
}

.chatgpt-brand-subtitle{
  font-size:0.68rem;
  color:#7180A2;
  text-transform:uppercase;
  letter-spacing:0.13em;
  margin-top:2px;
}

.chatgpt-section-title{
  font-family:'Space Grotesk',sans-serif;
  font-size:0.78rem;
  font-weight:600;
  color:#94A3B8;
  letter-spacing:0.04em;
  margin:18px 0 6px 2px;
}

.chatgpt-empty-history{
  display:flex;
  align-items:center;
  gap:8px;
  font-size:0.78rem;
  color:#687590;
  font-style:italic;
  padding:8px 2px;
}

.chatgpt-delete-box{
  font-size:0.78rem;
  color:#FCA5A5;
  margin:4px 2px 6px 2px;
}

.chatgpt-sidebar-spacer{ height:14px; }

.chatgpt-capability{
  display:flex;
  gap:10px;
  align-items:flex-start;
  padding:7px 0;
  font-size:0.8rem;
  color:#8E9BB8;
}

.chatgpt-capability b{ color:#E2E8F0; display:block; }
.chatgpt-capability small{ color:#8491AF; font-size:0.72rem; }

/* New chat button */
section[data-testid="stSidebar"] .st-key-sidebar_new_chat button{
  background:linear-gradient(135deg,#6366F1,#7C3AED) !important;
  border:1px solid rgba(129,140,248,0.40) !important;
  color:#FFFFFF !important;
  font-weight:700 !important;
  box-shadow:0 5px 18px rgba(99,102,241,0.25) !important;
}

section[data-testid="stSidebar"] .st-key-sidebar_new_chat button:hover{
  background:linear-gradient(135deg,#4F46E5,#6D28D9) !important;
}

/* History rows: flat list; the open chat is the "primary" button */
section[data-testid="stSidebar"] [class*="st-key-open_chat_"] button{
  justify-content:flex-start !important;
  text-align:left !important;
  background:transparent !important;
  border:1px solid transparent !important;
  color:#AAB5CD !important;
  font-weight:500 !important;
  box-shadow:none !important;
}

section[data-testid="stSidebar"] [class*="st-key-open_chat_"] button:hover{
  background:rgba(99,102,241,0.10) !important;
  border-color:rgba(99,102,241,0.18) !important;
  color:#E2E8F0 !important;
}

section[data-testid="stSidebar"] [class*="st-key-open_chat_"] button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,rgba(99,102,241,0.20),rgba(34,211,238,0.08)) !important;
  border:1px solid rgba(99,102,241,0.30) !important;
  color:#C7D2FE !important;
  font-weight:700 !important;
}

section[data-testid="stSidebar"] [class*="st-key-menu_chat_"] button{
  background:transparent !important;
  border:none !important;
  color:#64718C !important;
  box-shadow:none !important;
}

section[data-testid="stSidebar"] [class*="st-key-menu_chat_"] button:hover{
  color:#F87171 !important;
  background:rgba(239,68,68,0.10) !important;
}

/* Document rows in the sidebar */
.pearl-doc-row{
  display:flex;
  align-items:center;
  gap:8px;
  font-size:0.82rem;
  color:#E2E8F0;
  word-break:break-word;
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

def render_file_chips_html(files):
    """Little ChatGPT-style attachment cards shown above a user message."""
    if not files:
        return ""

    icons = {"ok": "📄", "warn": "⚠️", "err": "❌", "pending": "⏳"}
    chips = []
    for f in files:
        state = f.get("state", "ok")
        name = htmllib.escape(f.get("name", "file"))
        note = htmllib.escape(f.get("note", ""))
        chips.append(
            f'<div class="pearl-file-chip {state}">'
            f'<span class="ico">{icons.get(state, "📄")}</span>'
            f'<span class="meta"><span class="name">{name}</span>'
            f'<span class="note">{note}</span></span>'
            f'</div>'
        )
    return '<div class="pearl-files">' + "".join(chips) + "</div>"


def render_message_html(role, content, files=None):

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
        body_html = (
            "<p>" + htmllib.escape(content).replace("\n", "<br>") + "</p>"
            if content
            else ""
        )
        body_html = render_file_chips_html(files) + body_html

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


def render_thinking_html(message="Preparing a considered response&hellip;"):
    return f"""
    <div class="pearl-row assistant">
      <div class="pearl-avatar assistant"></div>
      <div class="pearl-thinking">{message}</div>
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

# ============================================================
# DOCUMENT LIBRARY HELPERS
#   Talks to Neon directly through vector_store (same in-process
#   approach as the agent, so no FastAPI server is needed).
# ============================================================

if "doc_uploader_key" not in st.session_state:
    st.session_state.doc_uploader_key = 0

if "doc_flash" not in st.session_state:
    st.session_state.doc_flash = []

if "doc_list_cache" not in st.session_state:
    st.session_state.doc_list_cache = None


def load_documents(force=False):
    """Read the document list from Neon once, then reuse it until it changes."""
    if force or st.session_state.doc_list_cache is None:
        try:
            st.session_state.doc_list_cache = {
                "docs": vector_store.list_documents(),
                "error": None,
            }
        except Exception as exc:
            st.session_state.doc_list_cache = {"docs": [], "error": str(exc)}
    return st.session_state.doc_list_cache


def format_size(num_bytes):
    if not num_bytes:
        return ""
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


ALLOWED_DOC_TYPES = ["pdf", "docx", "txt", "md", "csv"]

# Streamlit >= 1.43 can put a "+" attach button inside st.chat_input.
# Older versions fall back to the sidebar uploader.
CHAT_FILE_SUPPORT = "accept_file" in inspect.signature(st.chat_input).parameters


def ingest_files(files):
    """Save uploaded files into the Neon vector store.

    Returns one dict per file: {"name", "state" (ok/warn/err), "note"}.
    """
    results = []

    for f in files:
        try:
            with st.spinner(f"Processing {f.name}..."):
                result = vector_store.add_document(
                    filename=f.name,
                    data=f.getvalue(),
                    content_type=f.type,
                )
            if result["status"] == "duplicate":
                results.append(
                    {"name": f.name, "state": "warn", "note": "Already uploaded"}
                )
            else:
                results.append(
                    {
                        "name": f.name,
                        "state": "ok",
                        "note": f"Added to knowledge base \u00b7 {result['chunk_count']} sections",
                    }
                )
        except ValueError as exc:
            results.append({"name": f.name, "state": "err", "note": str(exc)})
        except Exception as exc:
            results.append(
                {"name": f.name, "state": "err", "note": f"Upload failed - {exc}"}
            )

    if results:
        load_documents(force=True)

    return results


def render_documents_panel():

    # Messages from the last delete / legacy upload (shown after the rerun)
    for kind, text in st.session_state.doc_flash:
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        else:
            st.error(text)
    st.session_state.doc_flash = []

    if CHAT_FILE_SUPPORT:
        st.caption(
            "Add files with the + button in the message box "
            "(PDF, DOCX, TXT, MD or CSV). Pearl answers from them."
        )
    else:
        # Fallback for old Streamlit versions: classic sidebar uploader
        st.caption(
            "Upload PDF, DOCX, TXT, MD or CSV. Pearl will answer questions "
            "from these documents. (Upgrade Streamlit to 1.43+ to attach "
            "files from the chat box with a + button.)"
        )

        uploaded_files = st.file_uploader(
            "Choose documents",
            type=ALLOWED_DOC_TYPES,
            accept_multiple_files=True,
            key=f"doc_uploader_{st.session_state.doc_uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded_files and st.button(
            "💾  Save to knowledge base",
            key="doc_save_btn",
            use_container_width=True,
        ):
            flash = []
            for r in ingest_files(uploaded_files):
                kind = {"ok": "success", "warn": "warning"}.get(r["state"], "error")
                flash.append((kind, f"{r['name']}: {r['note']}"))
            st.session_state.doc_flash = flash
            st.session_state.doc_uploader_key += 1  # clears the uploader
            st.rerun()

    st.markdown("**Your documents**")

    cache = load_documents()

    if cache["error"]:
        st.error(f"Document library unavailable: {cache['error']}")
        return

    if not cache["docs"]:
        st.caption("No documents uploaded yet.")
        return

    for doc in cache["docs"]:
        col_name, col_del = st.columns([5, 1])
        with col_name:
            details = " · ".join(
                x for x in [
                    f"{doc['page_count']} pages" if doc.get("page_count") else "",
                    format_size(doc.get("file_size")),
                ] if x
            )
            st.markdown(f"📄 {doc['filename']}")
            if details:
                st.caption(details)
        with col_del:
            if st.button("🗑", key=f"doc_del_{doc['id']}", help="Delete this document"):
                try:
                    vector_store.delete_document(doc["id"])
                    st.session_state.doc_flash = [("success", f"{doc['filename']} deleted.")]
                except Exception as exc:
                    st.session_state.doc_flash = [("error", f"Delete failed: {exc}")]
                load_documents(force=True)
                st.rerun()


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

    if st.button(
        "＋  New chat",
        key="sidebar_new_chat",
        use_container_width=True,
    ):
        new_conversation()
        st.rerun()

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
                <div class="chatgpt-empty-icon">✹</div>
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

            history_col, delete_col = st.columns([6, 1])

            with history_col:

                if st.button(
                    f"✹  {title}",
                    key=f"open_chat_{cid}",
                    type="primary" if is_active else "secondary",
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
    # DOCUMENTS (upload -> Neon vector DB -> chat answers)
    # --------------------------------------------------------

    with st.expander("📄  Documents", expanded=False):
        # Filled in further down, once any file attached in the chat box
        # has been processed, so the list is never out of date.
        docs_slot = st.container()

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
                <span>📄</span>
                <div>
                    <b>Document Q&amp;A</b>
                    <small>Ask questions about your uploaded files</small>
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

    with st.expander("⚙️ Settings", expanded=False):

        # --------------------------------------------------------
        # PEARL AI STATUS
        # --------------------------------------------------------

        backend_connected = False

        try:
            # Replace this with your real agent/backend check
            if agent is not None:
                backend_connected = True
        except Exception:
            backend_connected = False


        # --------------------------------------------------------
        # CONNECTED
        # --------------------------------------------------------

        if backend_connected:

            st.html("""
            <div class="pearl-status connected">
                <span class="status-dot"></span>
                <span>Pearl AI Connected</span>
            </div>

            <div class="pearl-backend-info connected-box">

                <div class="pearl-icon">
                    ✹
                </div>

                <div class="pearl-status-content">
                    <strong>Pearl AI is ready</strong>
                    <span>Backend connected successfully.</span>
                    <span class="ask-text">
                        You can start asking questions.
                    </span>
                </div>

            </div>
            """)


        # --------------------------------------------------------
        # OFFLINE
        # --------------------------------------------------------

        else:

            st.html("""
            <div class="pearl-status offline">
                <span class="status-dot"></span>
                <span>Pearl AI Offline</span>
            </div>

            <div class="pearl-backend-info offline-box">

                <div class="pearl-icon">
                    ⏳
                </div>

                <div class="pearl-status-content">
                    <strong>Waiting for Pearl AI...</strong>
                    <span>The backend is currently offline.</span>
                    <span class="wait-text">
                        Please wait while Pearl AI connects.
                    </span>
                </div>

            </div>
            """)
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
            render_message_html(m["role"], m["content"], m.get("files")) for m in active_conv["messages"]
        )
        st.markdown(history_html, unsafe_allow_html=True)


# ============================================================
# AGENT RUNNER  (streaming + automatic retry when the model is busy)
#   - Streaming: the answer appears word by word instead of
#     after a long silent wait.
#   - Retry: 503 "high demand" / 429 rate-limit errors from the
#     model provider are temporary, so we retry a few times with
#     a short back-off before showing an error.
# ============================================================

USE_STREAMING = True
MAX_ATTEMPTS = 4

RETRYABLE_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "OVERLOADED", "HIGH DEMAND", "TIMED OUT", "TIMEOUT",
)


def is_retryable(exc):
    text = str(exc).upper()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def friendly_error(exc):
    if is_retryable(exc):
        return (
            "Pearl's AI model is very busy right now (the model provider "
            "returned a temporary high-demand error, even after several "
            "retries). Please send your message again in a few seconds."
        )
    return f"An unexpected error occurred: {str(exc)}"


def chunk_text(content):
    """Visible text of one streamed chunk (ignores thinking/tool blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") in (None, "text")
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _run_streaming(payload, config, on_text, on_status):
    buffer = ""
    current_id = None
    tool_turn = False
    final_state = None
    last_paint = 0.0

    for mode, data in agent.stream(
        payload, config=config, stream_mode=["messages", "values"]
    ):
        if mode == "values":
            final_state = data
            continue

        if mode != "messages":
            continue

        chunk, meta = data

        # ignore sub-agent output and anything that is not model text
        if "|" in (meta or {}).get("langgraph_checkpoint_ns", ""):
            continue
        if getattr(chunk, "type", "") not in ("AIMessageChunk", "ai"):
            continue

        # a new model turn starts: forget text from the previous turn
        chunk_id = getattr(chunk, "id", None)
        if chunk_id != current_id:
            current_id = chunk_id
            buffer = ""
            tool_turn = False

        # this turn is calling a tool (search, documents...), not answering
        if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
            if not tool_turn:
                tool_turn = True
                on_status("Searching and working on your request&hellip;")
            continue
        if tool_turn:
            continue

        text = chunk_text(chunk.content)
        if text:
            buffer += text
            now = time.monotonic()
            if now - last_paint > 0.06:
                on_text(buffer)
                last_paint = now

    reply = extract_clean_text(final_state) if final_state is not None else ""
    return reply or buffer


def run_agent(prompt, thread_id, on_text, on_status):
    payload = {"messages": [{"role": "user", "content": prompt}]}
    config = {"configurable": {"thread_id": thread_id}}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if USE_STREAMING and hasattr(agent, "stream"):
                return _run_streaming(payload, config, on_text, on_status)
            return extract_clean_text(agent.invoke(payload, config=config))

        except Exception as exc:
            if attempt == MAX_ATTEMPTS or not is_retryable(exc):
                raise
            wait = 2 ** attempt  # 2s, 4s, 8s
            on_status(
                f"The model is busy &mdash; retrying in {wait}s "
                f"(attempt {attempt + 1} of {MAX_ATTEMPTS})&hellip;"
            )
            time.sleep(wait)


# ============================================================
# CHAT INPUT  (ChatGPT style: "+" button inside the message box)
# ============================================================

if CHAT_FILE_SUPPORT:
    submission = st.chat_input(
        "Ask anything...",
        accept_file="multiple",
        file_type=ALLOWED_DOC_TYPES,
        key="pearl_chat_input",
    )
else:
    submission = st.chat_input("Ask anything...")

prompt = ""
attached_files = []

if submission:
    if isinstance(submission, str):
        prompt = submission.strip()
    else:
        prompt = (getattr(submission, "text", "") or "").strip()
        attached_files = list(getattr(submission, "files", []) or [])

docs_rendered = False

if prompt or attached_files:

    # Auto-title this chat from its first message (or first file)
    if active_conv["title"] in ("New chat", "", None):
        title_source = prompt or (attached_files[0].name if attached_files else "")
        short_title = title_source.replace("\n", " ")
        active_conv["title"] = (short_title[:40] + "…") if len(short_title) > 40 else short_title

    with chat_container:

        # Show the user's message straight away, with the files marked
        # "uploading" until they are saved.
        user_placeholder = st.empty()

        pending_chips = [
            {"name": f.name, "state": "pending", "note": "Uploading…"}
            for f in attached_files
        ]
        user_placeholder.markdown(
            render_message_html("user", prompt, pending_chips), unsafe_allow_html=True
        )

        file_results = ingest_files(attached_files) if attached_files else []

        active_conv["messages"].append(
            {"role": "user", "content": prompt, "files": file_results}
        )
        user_placeholder.markdown(
            render_message_html("user", prompt, file_results), unsafe_allow_html=True
        )

        # Refresh the sidebar document list now that new files are saved
        with docs_slot:
            render_documents_panel()
        docs_rendered = True

        thinking_placeholder = st.empty()

        if not prompt:
            # Files only, no question: confirm and wait for the next message
            added = [r["name"] for r in file_results if r["state"] == "ok"]
            if added:
                names = ", ".join(f"**{n}**" for n in added)
                bot_reply = (
                    f"I've added {names} to the knowledge base. "
                    "What would you like to know about it?"
                )
            else:
                bot_reply = (
                    "I couldn't add the attached file(s) - see the status above. "
                    "Please try again."
                )

            thinking_placeholder.markdown(
                render_message_html("assistant", bot_reply), unsafe_allow_html=True
            )
            active_conv["messages"].append({"role": "assistant", "content": bot_reply})

        else:
            thinking_placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

            def paint_text(text):
                thinking_placeholder.markdown(
                    render_message_html("assistant", text + " ▌"),
                    unsafe_allow_html=True,
                )

            def paint_status(message):
                thinking_placeholder.markdown(
                    render_thinking_html(message), unsafe_allow_html=True
                )

            try:

                # Runs the deep agent in-process (no FastAPI). Each chat keeps
                # its own thread_id so separate chats keep separate memory.
                bot_reply = run_agent(
                    prompt,
                    st.session_state.active_id,
                    on_text=paint_text,
                    on_status=paint_status,
                )

                if not bot_reply:
                    bot_reply = "No response could be generated for this request."

                thinking_placeholder.markdown(
                    render_message_html("assistant", bot_reply), unsafe_allow_html=True
                )
                active_conv["messages"].append({"role": "assistant", "content": bot_reply})

            except Exception as e:
                error_message = friendly_error(e)
                thinking_placeholder.markdown(
                    render_error_html(error_message), unsafe_allow_html=True
                )
                active_conv["messages"].append({"role": "assistant", "content": error_message})

    # Redraw once so the sidebar chat title updates right after the first message
    st.rerun()

# No new submission this run: just draw the document list
if not docs_rendered:
    with docs_slot:
        render_documents_panel()

st.markdown(
        """
        <div class="pearl-credit">
            <span class="pearl-credit-line">Crafted by <b>Muthuraj</b></span>
            <span class="pearl-credit-line">&nbsp;•&nbsp;© 2026  Pearl Chat AI &nbsp;•&nbsp;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
