import streamlit as st
import streamlit.components.v1 as components
import json
import ast
import re
import copy
import html as htmllib
import uuid
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


/* ============================================================
   MESSAGE ACTION BAR
   ============================================================ */

[class*="st-key-act_"]{
  margin-top:-10px;
  margin-bottom:6px;
}

[class*="st-key-act_"] [data-testid="stHorizontalBlock"]{
  gap:0.15rem !important;
  align-items:center;
}

[class*="st-key-act_"] button{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  color:#7F8BA6 !important;
  padding:2px 8px !important;
  min-height:30px !important;
  font-size:0.95rem !important;
  border-radius:8px !important;
}

[class*="st-key-act_"] button:hover{
  background:rgba(148,163,184,0.14) !important;
  color:#E2E8F0 !important;
}

[class*="st-key-act_"] button[kind="primary"]{
  color:#A5B4FC !important;
  background:rgba(99,102,241,0.20) !important;
}

[class*="st-key-act_"] iframe{
  border:none !important;
  background:transparent !important;
}

.pearl-edit-note{
  font-size:0.75rem;
  color:#7F8BA6;
  margin:4px 0;
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
# MESSAGE ACTIONS
#   Copy / Read aloud / Share run in the browser (HTML component).
#   Edit / Try again / Rate / Branch / Versions are Streamlit
#   buttons that change the conversation in Python.
# ============================================================

ICONS = {
    "copy": '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/></svg>',
    "check": '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="#4ADE80" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    "share": '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V4M8 8l4-4 4 4"/><path d="M5 13v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"/></svg>',
    "speak": '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z"/><path d="M15.5 9a4 4 0 0 1 0 6"/><path d="M18 6.5a8 8 0 0 1 0 11"/></svg>',
    "stop": '<svg viewBox="0 0 24 24" width="17" height="17" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
}

TOOLTIPS = {"copy": None, "share": "Share", "speak": "Read aloud"}


def client_actions(text, actions, is_user):
    """Copy / Read aloud / Share — these run in the browser."""

    payload = json.dumps(text).replace("</", "<\\/")
    copy_tip = "Copy message" if is_user else "Copy response"

    buttons = ""
    for a in actions:
        tip = copy_tip if a == "copy" else TOOLTIPS[a]
        buttons += f'<button id="b-{a}" title="{tip}" onclick="act_{a}(this)">{ICONS[a]}</button>'

    html = f"""
    <style>
      body{{margin:0;background:transparent;display:flex;gap:2px;font-family:sans-serif}}
      button{{background:transparent;border:none;color:#7F8BA6;cursor:pointer;
              width:32px;height:30px;border-radius:8px;display:flex;
              align-items:center;justify-content:center;padding:0}}
      button:hover{{background:rgba(148,163,184,0.16);color:#E2E8F0}}
    </style>
    {buttons}
    <script>
      const T = {payload};
      const ICON = {json.dumps(ICONS)};

      function fallbackCopy(){{
        const ta = document.createElement('textarea');
        ta.value = T; document.body.appendChild(ta); ta.select();
        try {{ document.execCommand('copy'); }} catch(e) {{}}
        document.body.removeChild(ta);
      }}

      function flashCheck(btn){{
        btn.innerHTML = ICON.check;
        setTimeout(() => btn.innerHTML = ICON.copy, 1500);
      }}

      function act_copy(btn){{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(T).then(() => flashCheck(btn)).catch(() => {{ fallbackCopy(); flashCheck(btn); }});
        }} else {{ fallbackCopy(); flashCheck(btn); }}
      }}

      function act_share(btn){{
        if (navigator.share) {{
          navigator.share({{ text: T }}).catch(() => {{}});
        }} else {{
          act_copy(btn);
          btn.title = 'Copied to clipboard';
        }}
      }}

      function act_speak(btn){{
        const synth = window.speechSynthesis;
        if (!synth) return;
        if (synth.speaking) {{ synth.cancel(); btn.innerHTML = ICON.speak; return; }}
        const u = new SpeechSynthesisUtterance(T.replace(/[*_`#]/g, ''));
        u.onend = u.onerror = () => btn.innerHTML = ICON.speak;
        btn.innerHTML = ICON.stop;
        synth.speak(u);
      }}
    </script>
    """
    components.html(html, height=34)


def start_pending():
    st.session_state.pending = True


def apply_edit(conv, idx, new_text):
    """Save the old text + everything after it as a 'version', then re-ask."""
    msgs = conv["messages"]
    m = msgs[idx]
    m.setdefault("versions", []).append({
        "content": m["content"],
        "agent_text": m.get("agent_text", m["content"]),
        "tail": msgs[idx + 1:],
    })
    m["content"] = new_text
    m["agent_text"] = new_text
    conv["messages"] = msgs[: idx + 1]
    start_pending()


def switch_version(conv, idx, v_idx):
    """Swap the current branch with a stored version."""
    msgs = conv["messages"]
    m = msgs[idx]
    current = {
        "content": m["content"],
        "agent_text": m.get("agent_text", m["content"]),
        "tail": msgs[idx + 1:],
    }
    chosen = m["versions"][v_idx]
    m["versions"][v_idx] = current
    m["content"] = chosen["content"]
    m["agent_text"] = chosen["agent_text"]
    conv["messages"] = msgs[: idx + 1] + chosen["tail"]


def branch_chat(conv, idx):
    """Copy the chat up to this message into a brand-new conversation."""
    new_id = new_conversation()
    new_conv = st.session_state.conversations[new_id]
    new_conv["title"] = "Branch · " + conv["title"]
    new_conv["messages"] = [
        dict(copy.deepcopy(m), id=str(uuid.uuid4()))
        for m in conv["messages"][: idx + 1]
    ]


def render_actions(conv, idx, m):
    is_user = m["role"] == "user"
    mid = m["id"]

    with st.container(key=f"act_{mid}"):

        # ------------------ USER MESSAGE ACTIONS ------------------
        if is_user:
            has_versions = bool(m.get("versions"))
            cols = st.columns([12, 2.2, 1, 1] if has_versions else [12, 2.2, 1])

            with cols[1]:
                client_actions(m["content"], ["copy", "share"], True)

            with cols[2]:
                if st.button("✎", key=f"edit_{mid}", help="Edit message"):
                    st.session_state.editing = mid
                    st.rerun()

            if has_versions:
                with cols[3]:
                    with st.popover("⟲", help="See versions"):
                        st.caption("Other versions of this message")
                        options = [
                            f"{n + 1}. " + (v["content"][:40] + ("…" if len(v["content"]) > 40 else ""))
                            for n, v in enumerate(m["versions"])
                        ]
                        for n, label in enumerate(options):
                            if st.button(label, key=f"ver_{mid}_{n}", use_container_width=True):
                                switch_version(conv, idx, n)
                                st.rerun()
            return

        # ---------------- ASSISTANT MESSAGE ACTIONS ----------------
        cols = st.columns([0.9, 3.2, 1, 1, 1, 1, 9])

        with cols[1]:
            client_actions(m["content"], ["copy", "speak", "share"], False)

        with cols[2]:
            if st.button("👍", key=f"up_{mid}", help="Good response",
                         type="primary" if m.get("feedback") == "up" else "secondary"):
                m["feedback"] = None if m.get("feedback") == "up" else "up"
                st.rerun()

        with cols[3]:
            if st.button("👎", key=f"down_{mid}", help="Bad response",
                         type="primary" if m.get("feedback") == "down" else "secondary"):
                m["feedback"] = None if m.get("feedback") == "down" else "down"
                st.rerun()

        with cols[4]:
            if st.button("↻", key=f"retry_{mid}", help="Try again"):
                conv["messages"] = conv["messages"][:idx]   # drop this reply
                start_pending()
                st.rerun()

        with cols[5]:
            with st.popover("⋯", help="More actions"):
                t = m.get("time")
                if t:
                    st.caption(t.strftime("Today, %I:%M %p") if t.date() == datetime.now().date()
                               else t.strftime("%b %d, %I:%M %p"))
                if st.button("⑂  Branch in new chat", key=f"branch_{mid}", use_container_width=True):
                    branch_chat(conv, idx)
                    st.rerun()


def ask_agent(conv):
    """Send the whole visible conversation to the agent."""
    history = [
        {"role": m["role"], "content": m.get("agent_text", m["content"])}
        for m in conv["messages"]
        if not m.get("error")
    ]
    # A fresh thread_id per call, because the full history is sent each time.
    # This keeps Try again / Edit from duplicating turns in the agent's memory.
    result = agent.invoke(
        {"messages": history},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    return extract_clean_text(result) or "No response could be generated for this request."


# ============================================================
# DOCUMENT LIBRARY HELPERS
#   Talks to Neon directly through vector_store (same in-process
#   approach as the agent, so no FastAPI server is needed).
# ============================================================

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


def render_documents_panel():

    st.caption(
        "Attach documents with the + button in the chat box "
        "(PDF, DOCX, TXT, MD, CSV). Pearl answers from them."
    )

    # Messages from the last upload/delete (shown after the rerun)
    for kind, text in st.session_state.doc_flash:
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        else:
            st.error(text)
    st.session_state.doc_flash = []

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
    # DOCUMENTS (upload -> Neon vector DB -> chat answers)
    # --------------------------------------------------------

    with st.expander("📄  Documents", expanded=False):
        render_documents_panel()

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
# CHAT HISTORY (with action bars)
# ============================================================

if "pending" not in st.session_state:
    st.session_state.pending = False
if "editing" not in st.session_state:
    st.session_state.editing = None

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

    for i, m in enumerate(active_conv["messages"]):

        m.setdefault("id", str(uuid.uuid4()))   # also upgrades old messages

        # ---- Editing a user message ----
        if m["role"] == "user" and st.session_state.editing == m["id"]:
            st.markdown('<div class="pearl-edit-note">Editing message</div>', unsafe_allow_html=True)
            new_text = st.text_area(
                "Edit message", value=m["content"], key=f"edit_box_{m['id']}",
                label_visibility="collapsed",
            )
            c1, c2, _ = st.columns([1.3, 1, 6])
            with c1:
                if st.button("Save & send", key=f"save_{m['id']}", type="primary"):
                    st.session_state.editing = None
                    if new_text.strip():
                        apply_edit(active_conv, i, new_text.strip())
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"cancel_edit_{m['id']}"):
                    st.session_state.editing = None
                    st.rerun()
            continue

        st.markdown(render_message_html(m["role"], m["content"]), unsafe_allow_html=True)

        if not m.get("error"):
            render_actions(active_conv, i, m)

    # ---- Answer a pending question (new message, edit, or Try again) ----
    if st.session_state.pending:
        thinking = st.empty()
        thinking.markdown(render_thinking_html(), unsafe_allow_html=True)
        try:
            reply = ask_agent(active_conv)
            notes = st.session_state.pop("upload_notes", "")
            if notes:
                reply = notes + "\n\n" + reply
            active_conv["messages"].append({
                "id": str(uuid.uuid4()), "role": "assistant",
                "content": reply, "time": datetime.now(),
            })
        except Exception as e:
            active_conv["messages"].append({
                "id": str(uuid.uuid4()), "role": "assistant", "error": True,
                "content": f"An unexpected error occurred: {e}", "time": datetime.now(),
            })
        st.session_state.pending = False
        st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

DOC_FILE_TYPES = ["pdf", "docx", "txt", "md", "csv"]

chat_value = st.chat_input(
    "ASK Anything...  (use + to attach documents)",
    accept_file="multiple",
    file_type=DOC_FILE_TYPES,
)

if chat_value:

    # With accept_file enabled, the value carries both the typed text
    # and any attached files.
    prompt = (chat_value.get("text") or "").strip()
    attached_files = list(chat_value.get("files") or [])

    if prompt or attached_files:

        attached_names = [f.name for f in attached_files]

        # What the user sees in their own chat bubble
        user_display = prompt
        if attached_names:
            file_lines = "\n".join(f"📎 {name}" for name in attached_names)
            user_display = f"{prompt}\n\n{file_lines}" if prompt else file_lines

        # Auto-title this chat from its first message, like a history entry
        if active_conv["title"] in ("New chat", "", None):
            short_title = (prompt or attached_names[0]).replace("\n", " ")
            active_conv["title"] = (short_title[:40] + "…") if len(short_title) > 40 else short_title

        # ----------------------------------------------
        # 1) Save attached documents to Neon (vector DB)
        # ----------------------------------------------
        upload_notes, saved_names = [], []

        for f in attached_files:
            try:
                save_result = vector_store.add_document(
                    filename=f.name,
                    data=f.getvalue(),
                    content_type=f.type,
                )
                saved_names.append(f.name)

                if save_result["status"] == "duplicate":
                    upload_notes.append(f"ℹ️ **{f.name}** was already in your library.")
                else:
                    upload_notes.append(
                        f"✅ **{f.name}** saved ({save_result['chunk_count']} sections)."
                    )
            except ValueError as exc:
                upload_notes.append(f"❌ **{f.name}**: {exc}")
            except Exception as exc:
                upload_notes.append(f"❌ **{f.name}**: upload failed - {exc}")

        if attached_files:
            load_documents(force=True)

        # ----------------------------------------------
        # 2) Text the agent sees (hidden hint about attachments)
        # ----------------------------------------------
        agent_text = prompt
        if prompt and saved_names:
            agent_text += (
                "\n\n[The user just attached: "
                + ", ".join(saved_names)
                + ". They are now in the document library. "
                "Use search_documents to answer, and include the "
                "file name in your search query.]"
            )

        active_conv["messages"].append({
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": user_display,
            "agent_text": agent_text or user_display,
            "time": datetime.now(),
        })

        if prompt:
            # The agent runs at the top of the next rerun
            start_pending()
            if upload_notes:
                st.session_state.upload_notes = "\n".join(upload_notes)
        else:
            active_conv["messages"].append({
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "time": datetime.now(),
                "content": "\n".join(upload_notes)
                + "\n\nYou can now ask me questions about your document.",
            })

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
