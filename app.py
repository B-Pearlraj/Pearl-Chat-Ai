import streamlit as st
import json
import ast
import re
import html as htmllib
import uuid
import copy
from datetime import datetime
import os

from main import agent
import vector_store
import streamlit.components.v1 as components

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


</style>
"""

st.markdown(PEARL_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* ---------- Message action bars (copy, rate, read aloud...) ---------- */
    div[class*="st-key-msgbar_"]{
        margin-top:-14px;
        margin-bottom:4px;
    }
    div[class*="st-key-msgbar_a_"]{ padding-left:46px; }
    div[class*="st-key-msgbar_u_"]{ padding-right:46px; }

    div[class*="st-key-msgbar_"] [data-testid="stHorizontalBlock"]{
        flex-wrap:nowrap !important;
        gap:2px !important;
        align-items:center !important;
    }
    div[class*="st-key-msgbar_u_"] [data-testid="stHorizontalBlock"]{
        justify-content:flex-end !important;
    }
    div[class*="st-key-msgbar_"] [data-testid="stColumn"]{
        flex:0 0 auto !important;
        width:auto !important;
        min-width:0 !important;
    }
    div[class*="st-key-msgbar_"] .stButton > button,
    div[class*="st-key-msgbar_"] [data-testid="stPopover"] button{
        background:transparent !important;
        border:none !important;
        box-shadow:none !important;
        color:#9CA8C7 !important;
        height:32px !important;
        min-height:32px !important;
        width:32px !important;
        padding:0 !important;
        border-radius:8px !important;
    }
    div[class*="st-key-msgbar_"] [data-testid="stPopover"] button{
        width:auto !important;
        padding:0 8px !important;
    }
    div[class*="st-key-msgbar_"] .stButton > button:hover,
    div[class*="st-key-msgbar_"] [data-testid="stPopover"] button:hover{
        background:rgba(148,163,184,0.14) !important;
        color:#FFFFFF !important;
    }
    div[class*="st-key-rate_up_sel_"] button,
    div[class*="st-key-rate_down_sel_"] button{
        color:#A5B4FC !important;
        background:rgba(99,102,241,0.20) !important;
    }
    div[class*="st-key-msgbar_"] iframe{
        border:none !important;
        background:transparent !important;
    }
    div[class*="st-key-editbox_"]{
        margin:10px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# MESSAGE ACTIONS (copy, share, edit, rate, read aloud, retry, branch)
#   Copy / share / read aloud run in the browser (small iframe).
#   Edit / rate / retry / branch change the chat, so they are
#   normal Streamlit buttons with callbacks.
# ============================================================

ATTACH_PREFIX = "📎 "

_CLIENT_ACTIONS_HTML = r"""
<style>
  html,body{margin:0;padding:0;background:transparent;overflow:hidden;}
  body{display:flex;gap:2px;align-items:center;height:34px;}
  .b{width:32px;height:32px;border:none;border-radius:8px;background:transparent;
     color:#9CA8C7;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;}
  .b:hover{background:rgba(148,163,184,.14);color:#fff;}
  .b svg{width:17px;height:17px;}
</style>
__BUTTONS__
<script>
const TEXT = __TEXT__;
const ICONS = {
  copy:  '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  share: '<path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/>',
  speak: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',
  stop:  '<rect x="6" y="6" width="12" height="12" rx="2"/>'
};
function svg(k){
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+ICONS[k]+'</svg>';
}
function flash(btn, kind){
  btn.innerHTML = svg('check'); btn.style.color = '#34D399';
  setTimeout(function(){ btn.innerHTML = svg(kind); btn.style.color = ''; }, 1500);
}
function legacyCopy(t){
  var ta = document.createElement('textarea');
  ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  var ok = false;
  try { ok = document.execCommand('copy'); } catch(e) {}
  document.body.removeChild(ta);
  return ok;
}
function copyText(t){
  return new Promise(function(resolve){
    if (navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(t).then(function(){resolve(true);}).catch(function(){resolve(legacyCopy(t));});
    } else { resolve(legacyCopy(t)); }
  });
}
function plain(t){
  return t.replace(/```[\s\S]*?```/g,' code block. ').replace(/[*_`#>]/g,'').replace(/\s+/g,' ').trim();
}
function speak(btn){
  if (!('speechSynthesis' in window)) return;
  if (speechSynthesis.speaking){ speechSynthesis.cancel(); btn.innerHTML = svg('speak'); return; }
  var u = new SpeechSynthesisUtterance(plain(TEXT));
  u.lang = /[\u0B80-\u0BFF]/.test(TEXT) ? 'ta-IN' : 'en-US';
  u.onend = u.onerror = function(){ btn.innerHTML = svg('speak'); };
  btn.innerHTML = svg('stop');
  speechSynthesis.speak(u);
}
async function share(btn){
  if (navigator.share){
    try { await navigator.share({text: TEXT}); return; }
    catch(e){ if (e && e.name === 'AbortError') return; }
  }
  var ok = await copyText(TEXT);
  if (ok) flash(btn, 'share');
}
document.querySelectorAll('.b').forEach(function(b){
  var kind = b.dataset.a;
  b.innerHTML = svg(kind);
  b.addEventListener('click', function(){
    if (kind === 'copy'){ copyText(TEXT).then(function(ok){ if (ok) flash(b, 'copy'); }); }
    else if (kind === 'speak'){ speak(b); }
    else if (kind === 'share'){ share(b); }
  });
});
window.addEventListener('beforeunload', function(){ try { speechSynthesis.cancel(); } catch(e) {} });
</script>
"""


def strip_attachment_lines(text):
    """Remove the '📎 file.pdf' display lines from a stored user message."""
    lines = [ln for ln in (text or "").split("\n") if not ln.startswith(ATTACH_PREFIX)]
    return "\n".join(lines).strip()


def message_time_label(ts):
    if not ts:
        return ""
    clock = ts.strftime("%I:%M %p").lstrip("0")
    if ts.date() == datetime.now().date():
        return f"Today, {clock}"
    return f"{ts.strftime('%b')} {ts.day}, {clock}"


def render_client_actions(actions, text):
    """actions = [(kind, tooltip), ...] with kind in copy / share / speak."""
    buttons = "".join(
        f'<button class="b" data-a="{kind}" title="{htmllib.escape(tip)}" '
        f'aria-label="{htmllib.escape(tip)}"></button>'
        for kind, tip in actions
    )
    payload = json.dumps(text).replace("</", "<\\/")
    page = _CLIENT_ACTIONS_HTML.replace("__BUTTONS__", buttons).replace("__TEXT__", payload)
    components.html(page, width=34 * len(actions), height=34)


# ---------- callbacks (run before the next rerun) ----------

def cb_rate(cid, idx, value):
    msg = st.session_state.conversations[cid]["messages"][idx]
    msg["rating"] = None if msg.get("rating") == value else value


def cb_edit_start(cid, idx):
    st.session_state.editing = {"conv": cid, "idx": idx}


def cb_edit_cancel(cid, idx):
    st.session_state.editing = None
    st.session_state.pop(f"edit_text_{cid}_{idx}", None)


def cb_edit_send(cid, idx):
    conv = st.session_state.conversations[cid]
    new_text = (st.session_state.get(f"edit_text_{cid}_{idx}") or "").strip()
    if not new_text:
        return

    original = conv["messages"][idx]["content"]
    attach_lines = [ln for ln in original.split("\n") if ln.startswith(ATTACH_PREFIX)]
    display = new_text + ("\n\n" + "\n".join(attach_lines) if attach_lines else "")

    history = list(conv["messages"][:idx])
    conv["messages"] = history + [
        {"role": "user", "content": display, "time": datetime.now()}
    ]
    if idx == 0:
        conv["title"] = (new_text[:40] + "…") if len(new_text) > 40 else new_text

    # The agent's saved memory still contains the old wording, so start a
    # fresh agent thread and replay the earlier messages into it.
    conv["thread_id"] = str(uuid.uuid4())
    conv["replay"] = True

    st.session_state.pending_run = {"conv": cid, "text": new_text, "history": history}
    st.session_state.editing = None
    st.session_state.pop(f"edit_text_{cid}_{idx}", None)


def cb_retry(cid):
    conv = st.session_state.conversations[cid]
    msgs = conv["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        return
    msgs.pop()
    if not msgs or msgs[-1]["role"] != "user":
        return
    text = strip_attachment_lines(msgs[-1]["content"])
    if not text:
        return

    conv["thread_id"] = str(uuid.uuid4())
    conv["replay"] = True
    st.session_state.pending_run = {
        "conv": cid,
        "text": text,
        "history": list(msgs[:-1]),
    }


def cb_branch(cid, idx):
    conv = st.session_state.conversations[cid]
    kept = copy.deepcopy(conv["messages"][: idx + 1])
    old_title = conv.get("title") or "Chat"

    new_id = new_conversation()  # also makes it the active chat
    branch = st.session_state.conversations[new_id]
    branch["messages"] = kept
    branch["title"] = f"Branch · {old_title}"[:46]
    branch["thread_id"] = str(uuid.uuid4())
    branch["replay"] = True


def run_agent_reply(conv, text, history):
    """Call the agent. After an edit / retry / branch the conversation has a
    fresh thread, so the earlier messages are replayed once as context."""

    msgs = []

    if conv.get("replay"):
        for m in history:
            content = (
                strip_attachment_lines(m["content"])
                if m["role"] == "user"
                else m["content"]
            )
            if content:
                msgs.append({"role": m["role"], "content": content})
        while msgs and msgs[0]["role"] == "assistant":
            msgs.pop(0)
        conv["replay"] = False

    msgs.append({"role": "user", "content": text})

    result = agent.invoke(
        {"messages": msgs},
        config={
            "configurable": {
                "thread_id": conv.get("thread_id") or st.session_state.active_id,
            }
        },
    )

    reply = extract_clean_text(result)
    return reply or "No response could be generated for this request."


# ---------- rendering ----------

def render_message_actions(cid, conv, idx, msg, is_last):
    role = msg["role"]
    uid = f"{cid}_{idx}"

    # ---------------- user message ----------------
    if role == "user":
        clean = strip_attachment_lines(msg["content"])
        if not clean:
            return

        with st.container(key=f"msgbar_u_{uid}"):
            cols = st.columns(2)
            with cols[0]:
                render_client_actions(
                    [("copy", "Copy message"), ("share", "Share prompt")], clean
                )
            with cols[1]:
                st.button(
                    "",
                    key=f"edit_{uid}",
                    icon=":material/edit:",
                    help="Edit message",
                    on_click=cb_edit_start,
                    args=(cid, idx),
                )
        return

    # ---------------- assistant message ----------------
    text = msg["content"]
    rating = msg.get("rating")

    can_retry = False
    if is_last and idx > 0 and conv["messages"][idx - 1]["role"] == "user":
        can_retry = bool(strip_attachment_lines(conv["messages"][idx - 1]["content"]))

    with st.container(key=f"msgbar_a_{uid}"):
        cols = st.columns(6 if can_retry else 5)

        with cols[0]:
            render_client_actions([("copy", "Copy response")], text)
        with cols[1]:
            st.button(
                "",
                key=f"rate_up_sel_{uid}" if rating == "up" else f"rate_up_{uid}",
                icon=":material/thumb_up:",
                help="Rate response",
                on_click=cb_rate,
                args=(cid, idx, "up"),
            )
        with cols[2]:
            st.button(
                "",
                key=f"rate_down_sel_{uid}" if rating == "down" else f"rate_down_{uid}",
                icon=":material/thumb_down:",
                help="Rate response",
                on_click=cb_rate,
                args=(cid, idx, "down"),
            )
        with cols[3]:
            render_client_actions(
                [("speak", "Read aloud"), ("share", "Share")], text
            )

        next_col = 4
        if can_retry:
            with cols[next_col]:
                st.button(
                    "",
                    key=f"retry_{uid}",
                    icon=":material/refresh:",
                    help="Try again...",
                    on_click=cb_retry,
                    args=(cid,),
                )
            next_col += 1

        with cols[next_col]:
            with st.popover("⋯", help="More actions"):
                label = message_time_label(msg.get("time"))
                if label:
                    st.caption(label)
                st.button(
                    "Branch in new chat",
                    key=f"branch_{uid}",
                    icon=":material/call_split:",
                    on_click=cb_branch,
                    args=(cid, idx),
                )


def render_chat_history(cid, conv):
    editing = st.session_state.get("editing")
    messages = conv["messages"]

    for idx, m in enumerate(messages):

        is_editing = (
            editing
            and editing["conv"] == cid
            and editing["idx"] == idx
            and m["role"] == "user"
        )

        if is_editing:
            with st.container(key=f"editbox_{cid}_{idx}"):
                st.text_area(
                    "Edit message",
                    value=strip_attachment_lines(m["content"]),
                    key=f"edit_text_{cid}_{idx}",
                    height=120,
                    label_visibility="collapsed",
                )
                _, c_cancel, c_send = st.columns([6, 1.4, 1.4])
                with c_cancel:
                    st.button(
                        "Cancel",
                        key=f"edit_cancel_{cid}_{idx}",
                        on_click=cb_edit_cancel,
                        args=(cid, idx),
                        use_container_width=True,
                    )
                with c_send:
                    st.button(
                        "Send",
                        key=f"edit_send_{cid}_{idx}",
                        type="primary",
                        on_click=cb_edit_send,
                        args=(cid, idx),
                        use_container_width=True,
                    )
            continue

        if m["role"] == "assistant" and m["content"].startswith("An unexpected error occurred"):
            st.markdown(render_error_html(m["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                render_message_html(m["role"], m["content"]), unsafe_allow_html=True
            )

        render_message_actions(cid, conv, idx, m, idx == len(messages) - 1)


# ============================================================
# SIDEBAR — CHATGPT STYLE
# ============================================================

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
        render_chat_history(st.session_state.active_id, active_conv)


# ============================================================
# CHAT INPUT
# ============================================================

# ------------------------------------------------------------
# Re-run requested by Edit / Try again (agent answers again)
# ------------------------------------------------------------

pending_run = st.session_state.pop("pending_run", None)

if pending_run and pending_run["conv"] == st.session_state.active_id:

    with chat_container:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

        try:
            bot_reply = run_agent_reply(
                active_conv, pending_run["text"], pending_run["history"]
            )
        except Exception as e:
            bot_reply = f"An unexpected error occurred: {str(e)}"

        active_conv["messages"].append(
            {"role": "assistant", "content": bot_reply, "time": datetime.now()}
        )

    st.rerun()


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
            file_lines = "\n".join(f"{ATTACH_PREFIX}{name}" for name in attached_names)
            user_display = f"{prompt}\n\n{file_lines}" if prompt else file_lines

        history_before = list(active_conv["messages"])

        active_conv["messages"].append(
            {"role": "user", "content": user_display, "time": datetime.now()}
        )

        # Auto-title this chat from its first message, like a history entry
        if active_conv["title"] in ("New chat", "", None):
            short_title = (prompt or attached_names[0]).replace("\n", " ")
            active_conv["title"] = (short_title[:40] + "…") if len(short_title) > 40 else short_title

        with chat_container:
            st.markdown(render_message_html("user", user_display), unsafe_allow_html=True)

            thinking_placeholder = st.empty()

            thinking_placeholder.markdown(render_thinking_html(), unsafe_allow_html=True)

            try:

                # ----------------------------------------------
                # 1) Save attached documents to Neon (vector DB)
                # ----------------------------------------------
                upload_notes = []
                saved_names = []

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
                # 2) Ask the agent (only if the user typed a question)
                # ----------------------------------------------
                bot_reply = ""

                if prompt:
                    agent_text = prompt
                    if saved_names:
                        agent_text += (
                            "\n\n[The user just attached: "
                            + ", ".join(saved_names)
                            + ". They are now in the document library. "
                            "Use search_documents to answer, and include the "
                            "file name in your search query.]"
                        )

                    # Run the deep agent directly, in-process — no FastAPI
                    # server involved. Each conversation keeps its own
                    # thread so separate chats keep separate agent memory.
                    bot_reply = run_agent_reply(active_conv, agent_text, history_before)

                elif saved_names:
                    bot_reply = "You can now ask me questions about your document."

                if upload_notes:
                    bot_reply = "\n".join(upload_notes) + ("\n\n" + bot_reply if bot_reply else "")

                thinking_placeholder.markdown(
                    render_message_html("assistant", bot_reply), unsafe_allow_html=True
                )
                active_conv["messages"].append(
                    {"role": "assistant", "content": bot_reply, "time": datetime.now()}
                )

            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                thinking_placeholder.markdown(render_error_html(error_message), unsafe_allow_html=True)
                active_conv["messages"].append(
                    {"role": "assistant", "content": error_message, "time": datetime.now()}
                )

        # Redraw so the new messages get their action icons
        # (copy, rate, read aloud...) and the sidebar document list refreshes.
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
