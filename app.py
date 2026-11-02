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

import copy
import streamlit.components.v1 as components

from main import (
    agent,
    search_session_documents,
    build_history_messages,
    save_feedback,
)
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


/* ============================================================
   MESSAGE ACTION BARS
   (copy / share / edit / versions / rate / read aloud /
    try again / more -> branch in new chat)
   ============================================================ */

/* each message = bubble + its action bar, kept close together */
[class*="st-key-msgwrap_"]{
  gap:0 !important;
}

[class*="st-key-msgwrap_"] .pearl-row{
  margin:14px 0 2px 0;
}

[class*="st-key-actbar_asst_"]{ padding-left:40px; margin-top:2px; }
[class*="st-key-actbar_user_"]{ padding-right:0; margin-top:2px; }

/* keep every icon on ONE line, sized to its content (also on phones) */
[class*="st-key-actbar_"] [data-testid="stHorizontalBlock"]{
  flex-wrap:nowrap !important;
  gap:0 !important;
  align-items:center !important;
}

[class*="st-key-actbar_user_"] [data-testid="stHorizontalBlock"]{
  justify-content:flex-end !important;
}

[class*="st-key-actbar_asst_"] [data-testid="stHorizontalBlock"]{
  justify-content:flex-start !important;
}

[class*="st-key-actbar_"] [data-testid="stColumn"],
[class*="st-key-actbar_"] [data-testid="column"]{
  flex:0 0 auto !important;
  width:auto !important;
  min-width:0 !important;
}

/* flat icon buttons */
[class*="st-key-actbar_"] button{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  color:#9CA8C7 !important;
  width:34px !important;
  min-width:34px !important;
  height:32px !important;
  min-height:32px !important;
  padding:0 !important;
  border-radius:8px !important;
}

[class*="st-key-actbar_"] button:hover:not(:disabled){
  background:rgba(148,163,184,0.16) !important;
  color:#F1F5F9 !important;
}

[class*="st-key-actbar_"] button:disabled{
  opacity:0.35 !important;
}

/* an active rating (thumbs up / down) */
[class*="st-key-actbar_"] button[data-testid="stBaseButton-primary"]{
  color:#A5B4FC !important;
  background:rgba(99,102,241,0.20) !important;
}

/* icon-only buttons: hide the (empty) text label */
[class*="st-key-actbar_"] button [data-testid="stMarkdownContainer"]{
  display:none !important;
}

/* popover triggers: hide the little drop-down arrow if present */
[data-testid="stPopoverButton"] > * > [aria-hidden="true"]:not(:first-child){
  display:none !important;
}

/* transparent iframes (copy / read aloud / share icons) */
[class*="st-key-actbar_"] iframe{
  color-scheme:normal !important;
  background:transparent !important;
  border:0 !important;
}

.pearl-vcount{
  font-size:0.74rem;
  color:#9CA8C7;
  padding:0 2px;
  white-space:nowrap;
}

/* popover menus (More / See versions) */
[data-testid="stPopoverBody"]{
  background:#151D33 !important;
  border:1px solid rgba(148,163,184,0.22) !important;
  border-radius:12px !important;
  min-width:230px;
}

.pearl-pop-title{
  font-size:0.78rem;
  color:#9CA8C7;
  padding:2px 4px 6px 4px;
}

[class*="st-key-branch_"] button,
[class*="st-key-verpick_"] button{
  justify-content:flex-start !important;
  text-align:left !important;
}

/* inline edit box (replaces the bubble while editing) */
[class*="st-key-editbox_"]{
  padding-left:10%;
  padding-right:46px;
  margin:14px 0 6px 0;
}

[class*="st-key-editbox_"] textarea{
  background:#11182D !important;
  color:#E8ECF8 !important;
  border:1px solid rgba(99,102,241,0.45) !important;
  border-radius:14px !important;
  font-family:'Inter',sans-serif !important;
  font-size:0.92rem !important;
}

[class*="st-key-editbox_"] button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#6366F1,#7C3AED) !important;
  border:none !important;
  color:#FFFFFF !important;
  font-weight:700 !important;
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
        "thread_id": conv_id,   # agent memory thread (changes when history is rewritten)
        "seeded": True,         # False = the thread must be refilled from the screen
        "pending": False,       # True = last user message still needs an answer
        "regen": None,          # earlier replies kept by "Try again"
    }
    st.session_state.active_id = conv_id
    return conv_id


def reset_thread(conv):
    """Give the chat a fresh agent memory thread.

    Needed whenever the visible history is rewritten (edit, try again,
    switching versions, clearing, branching). The next question then
    replays the visible history into the new thread, so the model always
    sees exactly what the person sees on screen.
    """
    conv["thread_id"] = str(uuid.uuid4())
    conv["seeded"] = False


def normalize_conv(conv_id, conv):
    """Fill in the fields the message actions need (also for older chats)."""
    conv.setdefault("thread_id", conv_id)
    conv.setdefault("seeded", True)
    conv.setdefault("pending", False)
    conv.setdefault("regen", None)

    for m in conv["messages"]:
        m.setdefault("id", str(uuid.uuid4()))
        m.setdefault("time", conv.get("created") or datetime.now())
        if m.get("role") == "assistant":
            m.setdefault("variants", [m.get("content", "")])
            m.setdefault("variant_idx", 0)
            m.setdefault("ratings", {})


if "editing" not in st.session_state:
    st.session_state.editing = None


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
normalize_conv(st.session_state.active_id, active_conv)

# one-shot message (e.g. after rating a reply) shown at the top of the next run
_toast_text = st.session_state.pop("pending_toast", None)
if _toast_text:
    st.toast(_toast_text)


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
# MESSAGE ACTIONS  (ChatGPT-style toolbar under every message)
#
#   Your messages   : See versions · Copy message · Share prompt ·
#                     Edit message
#   Pearl's replies : Copy response · Rate response (👍 / 👎) ·
#                     Read aloud · Share · Try again · More
#                     (More -> time stamp + Branch in new chat)
#
#   Copy / Read aloud / Share need the browser (clipboard, speech,
#   share sheet), so they live in a tiny sandboxed HTML button strip.
#   Everything that changes the conversation (rate, edit, try again,
#   versions, branch) is a normal Streamlit button.
# ============================================================

ACTION_ICONS = {
    "copy": '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    "check": '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>',
    "share": '<svg viewBox="0 0 24 24"><path d="M4 12v7a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>',
    "speak": '<svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>',
    "stop": '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
}

ACTION_IFRAME_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
  body { display:flex; align-items:center; justify-content:flex-start; gap:0; height:36px; }
  button {
    width:34px; height:32px; border:0; border-radius:8px; padding:0;
    background:transparent; color:#9CA8C7; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    transition:background .15s ease, color .15s ease;
  }
  button:hover { background:rgba(148,163,184,.16); color:#F1F5F9; }
  button.on   { color:#A5B4FC; background:rgba(99,102,241,.20); }
  button.done { color:#4ADE80; }
  button:disabled { opacity:.35; cursor:not-allowed; }
  svg { width:18px; height:18px; fill:none; stroke:currentColor; stroke-width:1.9;
        stroke-linecap:round; stroke-linejoin:round; pointer-events:none; }
</style>
</head>
<body>
__BUTTONS__
<script>
(function () {
  const CFG = __CFG__;
  const ICONS = __ICONS__;
  const byId = (id) => document.getElementById(id);

  async function copyText(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (e) { /* fall through to the legacy method */ }
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (e) { return false; }
  }

  function flash(btn, title) {
    if (btn.dataset.busy) return;
    btn.dataset.busy = '1';
    const oldHtml = btn.innerHTML, oldTitle = btn.title;
    btn.innerHTML = ICONS.check;
    btn.classList.add('done');
    btn.title = title;
    setTimeout(function () {
      btn.innerHTML = oldHtml;
      btn.classList.remove('done');
      btn.title = oldTitle;
      delete btn.dataset.busy;
    }, 1600);
  }

  // ---------------- Copy ----------------
  const copyBtn = byId('copy');
  if (copyBtn) {
    copyBtn.onclick = async function () {
      const ok = await copyText(CFG.text);
      flash(copyBtn, ok ? 'Copied' : 'Copy failed');
    };
  }

  // ---------------- Share ----------------
  // Uses the device share sheet when the browser has one, otherwise
  // copies the text to the clipboard.
  const shareBtn = byId('share');
  if (shareBtn) {
    shareBtn.onclick = async function () {
      try {
        if (navigator.share) {
          await navigator.share({ title: 'Pearl.Ai', text: CFG.text });
          return;
        }
      } catch (e) {
        if (e && e.name === 'AbortError') return;
      }
      const ok = await copyText(CFG.text);
      flash(shareBtn, ok ? 'Copied to clipboard' : 'Share failed');
    };
  }

  // ---------------- Read aloud ----------------
  const speakBtn = byId('speak');
  if (speakBtn) {
    let W = window;
    try {
      // use the parent page's speech engine so the voice keeps going
      // (and can be stopped) even when Streamlit redraws this strip
      if (window.parent && window.parent.speechSynthesis && window.parent.document) {
        W = window.parent;
      }
    } catch (e) { W = window; }

    const synth = W.speechSynthesis;
    let showing = false;

    const isActive = function () {
      return !!synth && W.__pearlSpeakId === CFG.id && (synth.speaking || synth.pending);
    };

    const paint = function (on) {
      showing = on;
      speakBtn.innerHTML = on ? ICONS.stop : ICONS.speak;
      speakBtn.classList.toggle('on', on);
      speakBtn.title = on ? 'Stop' : CFG.labels.speak;
    };

    const chunksOf = function (text) {
      const parts = text.match(/[^.!?\n]+[.!?]*\s*/g) || [text];
      const out = [];
      let cur = '';
      parts.forEach(function (p) {
        if (cur && (cur + p).length > 220) { out.push(cur); cur = p; }
        else { cur += p; }
      });
      if (cur.trim()) out.push(cur);
      return out;
    };

    if (!synth) {
      speakBtn.disabled = true;
      speakBtn.title = 'Read aloud is not supported in this browser';
    } else {
      if (isActive()) paint(true);

      speakBtn.onclick = function () {
        if (isActive()) {
          W.__pearlSpeakId = null;
          synth.cancel();
          paint(false);
          return;
        }
        synth.cancel();
        W.__pearlSpeakId = CFG.id;
        const lang = /[\u0B80-\u0BFF]/.test(CFG.speech) ? 'ta-IN' : '';
        chunksOf(CFG.speech).forEach(function (chunk) {
          const u = new W.SpeechSynthesisUtterance(chunk);
          if (lang) u.lang = lang;
          synth.speak(u);
        });
        paint(true);
      };

      // keeps the icon right when speech ends by itself or another
      // message starts talking
      setInterval(function () {
        const active = isActive();
        if (active !== showing) paint(active);
      }, 400);
    }
  }
})();
</script>
</body>
</html>
"""


def strip_markdown_for_speech(text):
    """Plain text for the voice: no asterisks, hashes, links or code."""
    t = re.sub(r"```.*?```", " (code block omitted) ", text or "", flags=re.DOTALL)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"(\*\*|__|\*)", "", t)
    return t.strip()


def action_iframe(uid, text, buttons, labels):
    """A small strip of browser-side buttons: copy / speak / share."""
    button_html = "".join(
        f'<button id="{name}" type="button" '
        f'title="{htmllib.escape(labels[name], quote=True)}" '
        f'aria-label="{htmllib.escape(labels[name], quote=True)}">'
        f"{ACTION_ICONS[name]}</button>"
        for name in buttons
    )

    cfg = {
        "id": uid,
        "text": text,
        "speech": strip_markdown_for_speech(text) if "speak" in buttons else "",
        "labels": labels,
    }
    cfg_json = (
        json.dumps(cfg, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )

    page = (
        ACTION_IFRAME_TEMPLATE
        .replace("__BUTTONS__", button_html)
        .replace("__ICONS__", json.dumps(ACTION_ICONS))
        .replace("__CFG__", cfg_json)
    )

    width = 34 * len(buttons)

    # Newer Streamlit replaced components.html with st.iframe
    if hasattr(st, "iframe"):
        st.iframe(page, width=width, height=36)
    else:
        components.html(page, height=36, width=width)


# ------------------------------------------------------------
# Small widget helpers (fall back gracefully on older Streamlit)
# ------------------------------------------------------------

def icon_button(key, icon, help_text, fallback="•", **kwargs):
    try:
        return st.button("", key=key, icon=icon, help=help_text, **kwargs)
    except TypeError:
        return st.button(fallback, key=key, help=help_text, **kwargs)


def icon_popover(icon, help_text, fallback="⋯"):
    try:
        return st.popover("", icon=icon, help=help_text)
    except TypeError:
        return st.popover(fallback)


def menu_button(label, key, icon, **kwargs):
    try:
        return st.button(label, key=key, icon=icon, **kwargs)
    except TypeError:
        return st.button(label, key=key, **kwargs)


def format_msg_time(ts):
    """'Today, 4:25 PM' / 'Yesterday, 9:05 AM' / 'Oct 31, 5:03 PM'."""
    if not ts:
        return ""
    now = datetime.now()
    clock = ts.strftime("%I:%M %p").lstrip("0")
    if ts.date() == now.date():
        return f"Today, {clock}"
    if (now.date() - ts.date()).days == 1:
        return f"Yesterday, {clock}"
    return f"{ts.strftime('%b')} {ts.day}, {clock}"


def make_title(text):
    short = (text or "").replace("\n", " ").strip()
    return (short[:40] + "…") if len(short) > 40 else short


# ------------------------------------------------------------
# Building / changing messages
# ------------------------------------------------------------

def make_user_message(content, files=None):
    return {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": content,
        "files": files or [],
        "time": datetime.now(),
    }


def make_assistant_message(content, variants=None, ratings=None, msg_id=None):
    variants = list(variants) if variants else [content]
    return {
        "id": msg_id or str(uuid.uuid4()),
        "role": "assistant",
        "content": content,
        "variants": variants,
        "variant_idx": len(variants) - 1,
        "ratings": dict(ratings or {}),
        "time": datetime.now(),
    }


def history_payload(messages):
    """Visible chat -> clean list of {role, content} for the model."""
    return build_history_messages(
        [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in messages
            if m.get("content")
        ]
    )


def set_rating(conv_id, conv, idx, msg, rating):
    """👍 / 👎 on a reply. Clicking the same one again removes it."""
    vidx = msg.get("variant_idx", 0)
    ratings = msg.setdefault("ratings", {})
    new_value = None if ratings.get(vidx) == rating else rating

    if new_value is None:
        ratings.pop(vidx, None)
    else:
        ratings[vidx] = new_value

    prompt = ""
    for j in range(idx - 1, -1, -1):
        if conv["messages"][j]["role"] == "user":
            prompt = conv["messages"][j].get("content", "")
            break

    try:
        save_feedback(
            conversation_id=conv_id,
            message_id=msg["id"],
            rating=new_value,
            prompt=prompt,
            response=msg.get("content", ""),
        )
    except Exception:
        pass  # feedback logging must never break the chat

    st.session_state.pending_toast = {
        "up": "Thanks for the feedback!",
        "down": "Thanks - this helps Pearl improve.",
        None: "Feedback removed.",
    }[new_value]


def apply_edit(conv, idx, new_text):
    """Replace a user message, keep the old one as a version, answer again."""
    msgs = conv["messages"]
    msg = msgs[idx]

    versions = msg.get("versions")
    if not versions:
        versions = [{"content": msg["content"], "time": msg.get("time"), "tail": []}]
        msg["version_idx"] = 0

    current = msg.get("version_idx", 0)
    versions[current]["content"] = msg["content"]
    versions[current]["tail"] = msgs[idx + 1:]

    now = datetime.now()
    versions.append({"content": new_text, "time": now, "tail": []})

    msg["versions"] = versions
    msg["version_idx"] = len(versions) - 1
    msg["content"] = new_text
    msg["time"] = now

    del msgs[idx + 1:]

    if idx == 0:
        conv["title"] = make_title(new_text)

    conv["regen"] = None
    reset_thread(conv)
    conv["pending"] = True


def switch_version(conv, idx, target):
    """Show another version of an edited message (and the replies under it)."""
    msgs = conv["messages"]
    msg = msgs[idx]
    versions = msg.get("versions") or []
    current = msg.get("version_idx", 0)

    if target == current or not (0 <= target < len(versions)):
        return

    versions[current]["content"] = msg["content"]
    versions[current]["tail"] = msgs[idx + 1:]
    del msgs[idx + 1:]

    chosen = versions[target]
    msg["content"] = chosen.get("content", "")
    msg["time"] = chosen.get("time") or msg.get("time")
    msg["version_idx"] = target

    tail = chosen.get("tail") or []
    msgs.extend(tail)
    chosen["tail"] = []

    if idx == 0:
        conv["title"] = make_title(msg["content"])

    conv["regen"] = None
    reset_thread(conv)
    # a version that never got an answer is answered now
    conv["pending"] = not tail


def start_regenerate(conv):
    """Try again: drop the last reply but remember it as an earlier version."""
    msgs = conv["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        return

    last = msgs.pop()
    conv["regen"] = {
        "id": last.get("id"),
        "variants": list(last.get("variants") or [last.get("content", "")]),
        "ratings": dict(last.get("ratings") or {}),
    }
    reset_thread(conv)
    conv["pending"] = True


def change_variant(conv, msg, delta):
    """‹ 1/2 › - flip between the replies produced by Try again."""
    variants = msg.get("variants") or [msg["content"]]
    new_idx = max(0, min(len(variants) - 1, msg.get("variant_idx", 0) + delta))
    msg["variant_idx"] = new_idx
    msg["content"] = variants[new_idx]
    reset_thread(conv)


def branch_conversation(conv, idx):
    """Copy the chat up to (and including) message idx into a NEW chat."""
    new_id = str(uuid.uuid4())

    skip = {"versions", "version_idx", "variants", "variant_idx", "ratings", "id"}
    copied = []
    for m in conv["messages"][: idx + 1]:
        c = {k: copy.deepcopy(v) for k, v in m.items() if k not in skip}
        c["id"] = str(uuid.uuid4())
        if c["role"] == "assistant":
            c["variants"] = [c["content"]]
            c["variant_idx"] = 0
            c["ratings"] = {}
        copied.append(c)

    base = conv.get("title") or "New chat"
    if base.startswith("Branch · "):
        base = base[len("Branch · "):]

    st.session_state.conversations[new_id] = {
        "title": f"Branch · {base}",
        "messages": copied,
        "created": datetime.now(),
        "thread_id": new_id,
        "seeded": False,  # the first question replays the copied history
        "pending": False,
        "regen": None,
    }
    st.session_state.active_id = new_id
    st.session_state.editing = None


# ------------------------------------------------------------
# Rendering the toolbars
# ------------------------------------------------------------

def render_user_actions(conv_id, conv, idx, msg):
    uid = f"{conv_id[:8]}_{idx}_{msg['id'][:6]}"
    versions = msg.get("versions") or []
    has_text = bool(msg.get("content"))

    slots = []
    if len(versions) > 1:
        slots.append("versions")
    if has_text:
        slots += ["clip", "edit"]
    if not slots:
        return

    with st.container(key=f"actbar_user_{uid}"):
        cols = st.columns(len(slots))

        for slot, col in zip(slots, cols):
            with col:

                if slot == "versions":
                    with icon_popover(":material/history:", "See versions", "🕘"):
                        st.markdown(
                            '<div class="pearl-pop-title">Versions of this message</div>',
                            unsafe_allow_html=True,
                        )
                        current = msg.get("version_idx", 0)
                        for vi, v in enumerate(versions):
                            text = msg["content"] if vi == current else v.get("content", "")
                            text = text.replace("\n", " ")
                            label = f"{vi + 1}. " + (text[:38] + "…" if len(text) > 38 else text)
                            if st.button(
                                label,
                                key=f"verpick_{uid}_{vi}",
                                type="primary" if vi == current else "secondary",
                                use_container_width=True,
                            ):
                                switch_version(conv, idx, vi)
                                st.rerun()

                elif slot == "clip":
                    action_iframe(
                        f"{uid}_u",
                        msg["content"],
                        ["copy", "share"],
                        {"copy": "Copy message", "share": "Share prompt"},
                    )

                elif slot == "edit":
                    if icon_button(
                        f"edit_{uid}", ":material/edit:", "Edit message", fallback="✏️"
                    ):
                        st.session_state.editing = (conv_id, idx)
                        st.rerun()


def render_assistant_actions(conv_id, conv, idx, msg, is_last):
    uid = f"{conv_id[:8]}_{idx}_{msg['id'][:6]}"
    msgs = conv["messages"]

    variants = msg.get("variants") or [msg["content"]]
    vidx = msg.get("variant_idx", 0)
    rating = (msg.get("ratings") or {}).get(vidx)

    can_retry = (
        is_last
        and idx > 0
        and msgs[idx - 1]["role"] == "user"
        and bool(msgs[idx - 1].get("content"))
    )

    slots = []
    if len(variants) > 1:
        slots += ["prev", "count", "next"]
    slots += ["copy", "up", "down", "audio"]
    if can_retry:
        slots.append("retry")
    slots.append("more")

    with st.container(key=f"actbar_asst_{uid}"):
        cols = st.columns(len(slots))

        for slot, col in zip(slots, cols):
            with col:

                if slot == "prev":
                    if icon_button(
                        f"prev_{uid}", ":material/chevron_left:", "Previous response",
                        fallback="‹", disabled=vidx <= 0,
                    ):
                        change_variant(conv, msg, -1)
                        st.rerun()

                elif slot == "count":
                    st.markdown(
                        f'<span class="pearl-vcount">{vidx + 1}/{len(variants)}</span>',
                        unsafe_allow_html=True,
                    )

                elif slot == "next":
                    if icon_button(
                        f"next_{uid}", ":material/chevron_right:", "Next response",
                        fallback="›", disabled=vidx >= len(variants) - 1,
                    ):
                        change_variant(conv, msg, 1)
                        st.rerun()

                elif slot == "copy":
                    action_iframe(
                        f"{uid}_c", msg["content"], ["copy"], {"copy": "Copy response"}
                    )

                elif slot == "up":
                    if icon_button(
                        f"up_{uid}", ":material/thumb_up:", "Good response",
                        fallback="👍", type="primary" if rating == "up" else "secondary",
                    ):
                        set_rating(conv_id, conv, idx, msg, "up")
                        st.rerun()

                elif slot == "down":
                    if icon_button(
                        f"down_{uid}", ":material/thumb_down:", "Bad response",
                        fallback="👎", type="primary" if rating == "down" else "secondary",
                    ):
                        set_rating(conv_id, conv, idx, msg, "down")
                        st.rerun()

                elif slot == "audio":
                    action_iframe(
                        f"{uid}_a",
                        msg["content"],
                        ["speak", "share"],
                        {"speak": "Read aloud", "share": "Share"},
                    )

                elif slot == "retry":
                    if icon_button(
                        f"retry_{uid}", ":material/refresh:", "Try again...", fallback="↻"
                    ):
                        start_regenerate(conv)
                        st.rerun()

                elif slot == "more":
                    with icon_popover(":material/more_horiz:", "More actions", "⋯"):
                        st.markdown(
                            f'<div class="pearl-pop-title">'
                            f'{htmllib.escape(format_msg_time(msg.get("time")))}</div>',
                            unsafe_allow_html=True,
                        )
                        if menu_button(
                            "Branch in new chat",
                            key=f"branch_{uid}",
                            icon=":material/call_split:",
                            use_container_width=True,
                        ):
                            branch_conversation(conv, idx)
                            st.rerun()


def render_edit_box(conv_id, conv, idx, msg):
    """Inline editor that replaces the user's bubble while editing."""
    uid = f"{conv_id[:8]}_{idx}"

    with st.container(key=f"editbox_{uid}"):
        new_text = st.text_area(
            "Edit message",
            value=msg["content"],
            key=f"edit_text_{uid}",
            height=120,
            label_visibility="collapsed",
        )
        c_cancel, c_send, _ = st.columns([1, 1, 4])
        with c_cancel:
            cancel = st.button("Cancel", key=f"edit_cancel_{uid}", use_container_width=True)
        with c_send:
            send = st.button(
                "Send", key=f"edit_send_{uid}", type="primary", use_container_width=True
            )

    if cancel:
        st.session_state.editing = None
        st.rerun()

    if send:
        text = (new_text or "").strip()
        if not text:
            st.session_state.pending_toast = "A message can't be empty."
        elif text != msg["content"]:
            apply_edit(conv, idx, text)
        st.session_state.editing = None
        st.rerun()


def render_history(conv_id, conv):
    """Every message with its own toolbar."""
    msgs = conv["messages"]
    busy = bool(conv.get("pending"))
    editing = st.session_state.get("editing")
    last_idx = len(msgs) - 1

    for idx, msg in enumerate(msgs):
        with st.container(key=f"msgwrap_{conv_id[:8]}_{idx}"):

            if (
                msg["role"] == "user"
                and not busy
                and editing is not None
                and tuple(editing) == (conv_id, idx)
            ):
                render_edit_box(conv_id, conv, idx, msg)
                continue

            st.markdown(
                render_message_html(msg["role"], msg["content"], msg.get("files")),
                unsafe_allow_html=True,
            )

            if busy:
                continue  # no toolbar while a reply is being written

            if msg["role"] == "user":
                render_user_actions(conv_id, conv, idx, msg)
            else:
                render_assistant_actions(conv_id, conv, idx, msg, idx == last_idx)


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

# Files THIS visitor uploaded. Only these are searched for this visitor,
# and the library is never listed, so nobody sees anyone else's documents.
if "session_docs" not in st.session_state:
    st.session_state.session_docs = []


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
            if f.name not in st.session_state.session_docs:
                st.session_state.session_docs.append(f.name)

            if result["status"] == "duplicate":
                results.append(
                    {"name": f.name, "state": "warn", "note": "Already in knowledge base"}
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

    return results


def render_legacy_uploader():
    """Only used on old Streamlit (< 1.43) that has no + button in the chat box."""

    for kind, text in st.session_state.doc_flash:
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        else:
            st.error(text)
    st.session_state.doc_flash = []

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=ALLOWED_DOC_TYPES,
        accept_multiple_files=True,
        key=f"doc_uploader_{st.session_state.doc_uploader_key}",
        label_visibility="collapsed",
    )

    if uploaded_files and st.button(
        "💾  Add to this chat", key="doc_save_btn", use_container_width=True
    ):
        flash = []
        for r in ingest_files(uploaded_files):
            kind = {"ok": "success", "warn": "warning"}.get(r["state"], "error")
            flash.append((kind, f"{r['name']}: {r['note']}"))
        st.session_state.doc_flash = flash
        st.session_state.doc_uploader_key += 1
        st.rerun()


with st.sidebar:

    # --------------------------------------------------------
    # SIDEBAR HEADER
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="chatgpt-side-header">
            <div class="chatgpt-logo">
                <img src="Face-PTS.png" class="pearl-logo" />
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

    # Documents are attached from the + button in the message box and are
    # private to this visitor, so there is no shared document list here.
    if not CHAT_FILE_SUPPORT:
        with st.expander("📎  Attach files", expanded=False):
            render_legacy_uploader()

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
        active_conv["pending"] = False
        active_conv["regen"] = None
        reset_thread(active_conv)
        st.session_state.editing = None

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

chat_container = st.container()   # filled further down (after the helpers)


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
    text = str(exc).upper()
    if "NOT_FOUND" in text and "MODEL" in text:
        return (
            "The configured Gemini model is not available for this API key "
            "(it may have been retired). Update GEMINI_MODEL or "
            "GEMINI_FALLBACK_MODELS in your environment settings."
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


def run_agent(prompt, thread_id, on_text, on_status, allowed_docs=None, history=None):
    # `history` (list of {role, content}) is only given for a brand-new thread
    # whose earlier messages must be replayed (after edit / try again / branch).
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    payload = {"messages": messages}
    config = {"configurable": {"thread_id": thread_id}}
    if allowed_docs is not None:
        config["configurable"]["allowed_docs"] = list(allowed_docs)

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
# DOCUMENTS FIRST
#   When the visitor has uploaded files, the best matching passages are
#   fetched here, before the model is called, and handed to it with the
#   question. The model answers from them instead of going to the web,
#   and it needs one model call instead of two (no separate tool step).
# ============================================================

def build_agent_prompt(prompt):
    docs = list(st.session_state.session_docs)

    if not docs:
        return prompt

    names = ", ".join(docs)

    try:
        passages = search_session_documents(prompt, docs, top_k=5)
    except Exception:
        passages = []

    if not passages:
        return (
            f"[DOCUMENT CONTEXT] The user uploaded these documents in this chat: "
            f"{names}. No relevant passage was found for the question below. "
            f"If the question is about their documents, say it was not found in "
            f"them; otherwise answer normally.\n[END DOCUMENT CONTEXT]\n\n"
            f"User question: {prompt}"
        )

    blocks = []
    for i, passage in enumerate(passages, 1):
        text = passage if isinstance(passage, str) else json.dumps(
            passage, ensure_ascii=False, default=str
        )
        blocks.append(f"Passage {i}: {text[:900]}")

    return (
        f"[DOCUMENT CONTEXT] Passages retrieved from the user's uploaded "
        f"documents ({names}). Answer from them when they contain the answer "
        f"and cite (Source: file, page). Do not use web_search for this "
        f"question unless the user explicitly asks for web or current "
        f"information. If the passages do not contain the answer, say it was "
        f"not found in the uploaded documents.\n"
        + "\n".join(blocks)
        + f"\n[END DOCUMENT CONTEXT]\n\nUser question: {prompt}"
    )


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

# ============================================================
# GENERATING A REPLY  (shared by: new message, edit, try again,
# switching to a version that has no answer yet)
# ============================================================

def generate_reply(conv, prompt, placeholder):
    """Run the agent and stream the answer into `placeholder`.

    Returns (text, is_error). The last message of `conv` must be the
    user's message that is being answered.
    """
    placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

    def paint_text(text):
        placeholder.markdown(
            render_message_html("assistant", text + " ▌"),
            unsafe_allow_html=True,
        )

    def paint_status(message):
        placeholder.markdown(render_thinking_html(message), unsafe_allow_html=True)

    # A chat whose history was rewritten (edit / try again / branch) has a
    # fresh, empty agent thread: replay what is on screen into it first.
    history = None
    if not conv.get("seeded", True):
        history = history_payload(conv["messages"][:-1])

    try:
        if st.session_state.session_docs:
            paint_status("Searching your documents&hellip;")

        reply = run_agent(
            build_agent_prompt(prompt),
            conv["thread_id"],
            on_text=paint_text,
            on_status=paint_status,
            allowed_docs=st.session_state.session_docs,
            history=history,
        )

        if not reply:
            reply = "No response could be generated for this request."

        conv["seeded"] = True
        return reply, False

    except Exception as exc:
        reset_thread(conv)  # the thread may be half-written: start clean next time
        return friendly_error(exc), True


def run_pending_generation(conv):
    """Answer the last user message (after an edit / try again / version switch)."""
    prompt = conv["messages"][-1].get("content", "")

    placeholder = st.empty()
    reply, is_error = generate_reply(conv, prompt, placeholder)

    placeholder.markdown(
        render_error_html(reply) if is_error else render_message_html("assistant", reply),
        unsafe_allow_html=True,
    )

    regen = conv.get("regen")
    if regen:
        # "Try again": keep earlier replies so ‹ 1/2 › can flip between them
        conv["messages"].append(
            make_assistant_message(
                reply,
                variants=regen["variants"] + [reply],
                ratings=regen["ratings"],
                msg_id=regen.get("id"),
            )
        )
    else:
        conv["messages"].append(make_assistant_message(reply))

    conv["regen"] = None
    conv["pending"] = False


# ============================================================
# DRAW THE CONVERSATION
#   (chat_container was reserved under the header above)
# ============================================================

new_submission = bool(prompt or attached_files)
empty_placeholder = None

with chat_container:

    if not active_conv["messages"]:
        empty_placeholder = st.empty()
        empty_placeholder.markdown(
            """
            <div class="pearl-empty">
              <div class="pearl-orb-hero"></div>
              <h3>Start a new conversation</h3>
              <div>Begin below with a question, a technical issue, or a request that requires current information.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        active_conv["pending"] = False

    else:
        render_history(st.session_state.active_id, active_conv)

        if active_conv.get("pending"):
            if active_conv["messages"][-1]["role"] == "user" and not new_submission:
                run_pending_generation(active_conv)
                st.rerun()
            else:
                active_conv["pending"] = False
                active_conv["regen"] = None


# ============================================================
# NEW MESSAGE FROM THE CHAT BOX
# ============================================================

if prompt or attached_files:

    active_conv["pending"] = False
    active_conv["regen"] = None
    st.session_state.editing = None

    # Auto-title this chat from its first message (or first file)
    if active_conv["title"] in ("New chat", "", None):
        title_source = prompt or (attached_files[0].name if attached_files else "")
        active_conv["title"] = make_title(title_source)

    with chat_container:

        if empty_placeholder is not None:
            empty_placeholder.empty()

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

        active_conv["messages"].append(make_user_message(prompt, file_results))
        user_placeholder.markdown(
            render_message_html("user", prompt, file_results), unsafe_allow_html=True
        )

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
            active_conv["messages"].append(make_assistant_message(bot_reply))

        else:
            # Runs the deep agent in-process (no FastAPI). Each chat keeps
            # its own thread_id so separate chats keep separate memory.
            bot_reply, reply_failed = generate_reply(active_conv, prompt, thinking_placeholder)

            thinking_placeholder.markdown(
                render_error_html(bot_reply)
                if reply_failed
                else render_message_html("assistant", bot_reply),
                unsafe_allow_html=True,
            )
            active_conv["messages"].append(make_assistant_message(bot_reply))

    # Redraw once so the sidebar chat title and the message toolbars update
    st.rerun()


st.markdown(
        """
        <div class="pearl-credit">
            <span class="pearl-credit-line">Crafted by <b>Muthuraj</b></span>
            <span class="pearl-credit-line">&nbsp;•&nbsp;© 2026  Pearl Chat AI &nbsp;•&nbsp;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
