import asyncio
import html
import json
import os
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination
)
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================
# st.html renders markup verbatim. st.markdown would treat
# indented HTML as a Markdown code block and show raw tags.

st.html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --brand-1: #6366f1;
    --brand-2: #8b5cf6;
    --brand-3: #ec4899;
    --ink: #1e1b4b;
    --muted: #4b5563;
    --line: #e5e7f5;
    --card: #ffffff;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif,
        'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji';
}

/* ---------- App shell ---------- */

/* One fixed pastel gradient behind everything. All layers above it
   (main column, header, bottom bar) are transparent so there are no
   seams. Layers are selected by data-testid / structure because
   Streamlit's emotion class names (e.g. st-emotion-cache-6px8kg)
   are regenerated between versions. */
.stApp {
    background: linear-gradient(
        160deg,
        #d9d2ff 0%,
        #d6e4ff 35%,
        #d3efe3 70%,
        #ffdccd 100%
    ) fixed;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div:has(> [data-testid="stMain"]),
[data-testid="stMain"],
[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

.block-container {
    max-width: 860px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

/* Bottom bar that holds the chat input */
[data-testid="stBottom"],
[data-testid="stBottom"] *:not(button):not(button *),
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}

[data-testid="stChatInput"] {
    border-radius: 16px;
    border: 1px solid #e4dcff;
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.10);
}

/* Drop the theme's grey inner box so only the card above shows */
[data-testid="stChatInput"] > div {
    border: 1px solid #AAAAAA !important;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
    color: #fff;
    border-radius: 10px;
}

[data-testid="stChatInput"] button:disabled {
    background: #ede9fe;
    color: #a78bfa;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--brand-2);
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.15);
}

/* ---------- Hero header ---------- */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--brand-1), var(--brand-2) 55%, var(--brand-3));
    border-radius: 24px;
    padding: 30px 32px;
    color: #fff;
    box-shadow: 0 18px 40px rgba(99, 102, 241, 0.25);
    margin-bottom: 1.5rem;
}

.hero::after {
    content: "";
    position: absolute;
    right: -60px;
    top: -60px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.12);
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.18);
    padding: 5px 12px;
    border-radius: 999px;
    margin-bottom: 14px;
}

.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.3);
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 8px;
}

.hero-subtitle {
    font-size: 15px;
    opacity: 0.92;
    max-width: 560px;
    line-height: 1.55;
}

.agent-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.agent-chip {
    font-size: 13px;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    padding: 6px 12px;
    border-radius: 999px;
}

/* ---------- Welcome / empty state ---------- */

.welcome {
    text-align: center;
    margin: 0.5rem 0 1.25rem;
}

.welcome-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 4px;
}

.welcome-text {
    color: var(--muted);
    font-size: 15px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 1.5rem;
}

.feature-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(30, 27, 75, 0.05);
}

.feature-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-size: 20px;
    margin-bottom: 10px;
}

.feature-title {
    font-weight: 650;
    color: var(--ink);
    font-size: 15px;
    margin-bottom: 4px;
}

.feature-text {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}

@media (max-width: 640px) {
    .feature-grid { grid-template-columns: 1fr; }
    .hero { padding: 24px; }
    .hero-title { font-size: 26px; }
}

.section-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}

/* Suggested-question buttons */
[class*="st-key-suggest-"] button {
    width: 100%;
    min-height: 64px;
    text-align: left;
    justify-content: flex-start;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    color: var(--ink);
    font-size: 14px;
    transition: all 0.15s ease;
}

[class*="st-key-suggest-"] button:hover {
    border-color: var(--brand-2);
    color: var(--brand-1);
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.12);
}

/* ---------- Chat messages ---------- */

[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.5rem 0;
}

/* User messages as a soft bubble */
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #d4ccff;
    border-radius: 16px;
    padding: 10px 16px;
}

/* Final answer card */
[class*="st-key-answer-"] {
    background: var(--card);
    border: 1px solid #ddd6fe !important;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 10px 28px rgba(124, 58, 237, 0.08);
}

.answer-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #5b21b6;
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    padding: 4px 10px;
    border-radius: 999px;
}

.guard-note {
    font-size: 12.5px;
    color: #92400e;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 6px 10px;
    margin-top: 8px;
}

.saved-note {
    font-size: 12.5px;
    color: #047857;
    margin-top: 6px;
}

/* Expanders + status */
[data-testid="stExpander"] details {
    border-radius: 14px;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.9);
}

[data-testid="stExpander"] summary p { font-weight: 600; }

/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}

.side-brand {
    font-size: 18px;
    font-weight: 800;
    color: var(--ink);
    margin-bottom: 2px;
}

.side-muted {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 18px;
}

.pipeline-step {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    position: relative;
}

.pipeline-step:not(:last-child)::after {
    content: "";
    position: absolute;
    left: 17px;
    top: 48px;
    bottom: -6px;
    width: 2px;
    background: var(--line);
}

.step-icon {
    flex: 0 0 36px;
    height: 36px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    font-size: 17px;
}

.step-title {
    font-weight: 600;
    font-size: 14px;
    color: var(--ink);
}

.step-text {
    font-size: 12.5px;
    color: var(--muted);
    line-height: 1.45;
}

.tint-indigo { background: #eef2ff; }
.tint-sky { background: #e0f2fe; }
.tint-pink { background: #fce7f3; }
.tint-amber { background: #fef3c7; }
</style>
"""
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


if not api_key:
    st.error("OPENAI_API_KEY is missing from your .env file.")
    st.stop()


if not serper_api_key:
    st.error("SERPER_API_KEY is missing from your .env file.")
    st.stop()


# Plain OpenAI client, used by the guardrails' moderation checks

@st.cache_resource
def get_openai_client() -> OpenAI:

    return OpenAI(api_key=api_key)


openai_client = get_openai_client()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# GREETING DETECTION
# ============================================================

def detect_greeting(query: str):

    normalized = query.lower().strip()

    normalized = re.sub(
        r"[!,.?]+$",
        "",
        normalized
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    )

    about_me = (
        "I'm an AI customer support assistant 🤖. Behind the scenes I "
        "coordinate three agents: a Knowledge Assistant, a Web Search "
        "Assistant and an Entry Agent that combines their answers into "
        "one reply. Ask me any support question to get started!"
    )

    capabilities = (
        "I can help with customer support questions such as orders, "
        "shipping, refunds, returns, accounts and passwords, billing "
        "and general product questions. Just type your question below."
    )

    greetings = {

        "hi":
            "Hello! 👋 How can I help you today?",

        "hello":
            "Hello! 👋 How can I help you today?",

        "hey":
            "Hey! 👋 How can I help you today?",

        "good morning":
            "Good morning! ☀️ How can I help you?",

        "good afternoon":
            "Good afternoon! 🌤️ How can I help you?",

        "good evening":
            "Good evening! 🌆 How can I help you?",

        "good night":
            "Good night! 🌙 If you have a question, I'm happy to help.",

        "who are you": about_me,
        "what are you": about_me,
        "who am i talking to": about_me,
        "are you a bot": about_me,
        "are you human": about_me,

        "what can you do": capabilities,
        "what do you do": capabilities,
        "how can you help": capabilities,
        "how can you help me": capabilities,
        "help": capabilities,

        "how are you":
            "I'm doing great, thanks for asking! 😊 How can I help you?",

        "thanks":
            "You're welcome! 😊 Anything else I can help with?",
        "thank you":
            "You're welcome! 😊 Anything else I can help with?",
        "thx":
            "You're welcome! 😊 Anything else I can help with?",

        "bye":
            "Goodbye! 👋 Come back anytime you need help.",
        "goodbye":
            "Goodbye! 👋 Come back anytime you need help."
    }

    # Too vague to be worth a multi-agent run: ask for detail instead.

    vague_inputs = {
        "", "what", "why", "how", "who", "when", "where", "which",
        "huh", "hmm", "hm", "ok", "okay", "k", "yes", "no", "yeah",
        "yep", "nope", "sure", "test", "testing", "lol", "anything"
    }

    if normalized in vague_inputs:

        return (
            "Could you tell me a bit more? 🙂 For example: *\"How do I "
            "return an item?\"* or *\"Where is my order?\"* The more "
            "detail you share, the better I can help."
        )

    return greetings.get(normalized)



# ============================================================
# GUARDRAILS - CONFIGURATION
# ============================================================

MAX_INPUT_CHARS = 2000

MODERATION_MODEL = "omni-moderation-latest"


# Phrases typical of attempts to override the agents' instructions.

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|system)\b.{0,20}\b(instructions?|prompts?|rules|messages?|guidelines)\b",
    r"\b(reveal|show|print|repeat|leak|tell me)\b.{0,30}\b(system|hidden|initial|original)\s+(prompt|message|instructions?)\b",
    r"\byou are now\b",
    r"\b(developer|god|jailbreak|dan)\s+mode\b",
    r"\bpretend (you are|to be)\b.{0,40}\b(unrestricted|unfiltered|without rules)\b",
    r"</?\s*(system|assistant)\s*>",
]


# Sensitive data. Order matters: earlier patterns win, so the
# context-based account pattern runs first and cards run before
# phone numbers (otherwise their digits get mislabelled).
#
# When a pattern has a (?P<secret>...) group, only that group is
# redacted and the surrounding context ("account no", "password is")
# is kept so the agents still understand the question.

PII_PATTERNS = [
    (
        "account number",
        r"(?i)\b(?:bank\s+)?(?:account|acct|a/c)\.?\s*(?:no\.?|number|num|#)?"
        r"\s*(?:is|:|-|=)?\s*(?P<secret>\d(?:[\s-]?\d){5,19})\b"
    ),
    ("card number", r"\b\d(?:[ -]?\d){12,18}\b"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("API key", r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),
    ("password or PIN", r"(?i)\b(?:password|passcode|pwd|pin|otp|cvv)\b\s*(?:is|:|=)\s*(?P<secret>\S+)"),
    ("email address", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Phones need separators ("415-555-2671", "+1 (415) 555 2671") so
    # bare digit runs such as order or tracking numbers are kept.
    ("phone number", r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)"),
]

# Output keeps emails/phones: support answers legitimately contain
# contact details such as support@company.com or a hotline number.

OUTPUT_REDACT = {"account number", "card number", "SSN", "API key", "password or PIN"}


# Fragments of the agents' system messages. If they appear in an
# answer, the model is leaking its instructions.

SYSTEM_PROMPT_MARKERS = [
    "you are the first customer support assistant",
    "you are the web search customer support assistant",
    "you are the final customer support and reconciliation agent",
    "call save_to_file with",
    "write terminate as the very last word",
]


# ============================================================
# GUARDRAILS - RESULT TYPE
# ============================================================

@dataclass
class GuardrailResult:

    allowed: bool

    # Sanitized text to use downstream (redacted if needed)
    text: str

    # User-facing explanation when blocked
    reason: str | None = None

    # Non-blocking notices, e.g. "Removed a card number"
    notes: list[str] = field(default_factory=list)


# ============================================================
# GUARDRAILS - HELPERS
# ============================================================

def _luhn_valid(digits: str) -> bool:

    total = 0

    for i, d in enumerate(reversed(digits)):

        n = int(d)

        if i % 2 == 1:

            n *= 2

            if n > 9:
                n -= 9

        total += n

    return total % 10 == 0


def redact_pii(text: str, kinds: set[str] | None = None) -> tuple[str, list[str]]:

    found = []

    for kind, pattern in PII_PATTERNS:

        if kinds is not None and kind not in kinds:
            continue

        def replace(match, kind=kind):

            value = match.group(0)

            # Only treat digit runs as cards if they pass Luhn,
            # so order and tracking numbers are left alone.

            if kind == "card number":

                digits = re.sub(r"\D", "", value)

                if not (13 <= len(digits) <= 19 and _luhn_valid(digits)):
                    return value

            found.append(kind)

            placeholder = f"[REDACTED {kind.upper()}]"

            if "secret" in match.re.groupindex:

                start, end = match.span("secret")

                offset = match.start()

                return (
                    value[:start - offset]
                    + placeholder
                    + value[end - offset:]
                )

            return placeholder

        text = re.sub(pattern, replace, text)

    return text, sorted(set(found))


def detect_injection(text: str) -> bool:

    lowered = text.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in INJECTION_PATTERNS
    )


def moderate(client: OpenAI, text: str) -> list[str] | None:
    """
    Return the flagged categories (empty list if clean), or None when
    the moderation API is unavailable. Callers fail open on None so
    an outage does not take the whole support chat down.
    """

    try:

        response = client.moderations.create(
            model=MODERATION_MODEL,
            input=text
        )

    except Exception:

        return None

    result = response.results[0]

    if not result.flagged:
        return []

    categories = result.categories.model_dump()

    return [
        name.replace("_", " ").replace("/", " / ")
        for name, flagged in categories.items()
        if flagged
    ]


# ============================================================
# INPUT GUARDRAIL
# ============================================================

def check_input(client: OpenAI, query: str) -> GuardrailResult:

    text = query.strip()

    if len(text) > MAX_INPUT_CHARS:

        return GuardrailResult(
            allowed=False,
            text=text,
            reason=(
                f"Your message is {len(text):,} characters long. Please "
                f"keep questions under {MAX_INPUT_CHARS:,} characters, "
                "or split it into smaller questions."
            )
        )

    if detect_injection(text):

        return GuardrailResult(
            allowed=False,
            text=text,
            reason=(
                "I can't change how I work or share my internal "
                "instructions, but I'm happy to help with a support "
                "question. 🙂"
            )
        )

    flagged = moderate(client, text)

    if flagged:

        return GuardrailResult(
            allowed=False,
            text=text,
            reason=(
                "I can't help with that request. If you have a "
                "customer support question, I'm glad to help."
            ),
            notes=[f"Flagged: {', '.join(flagged)}"]
        )

    notes = []

    if flagged is None:
        notes.append("Content moderation was unavailable for this message.")

    redacted, found = redact_pii(text)

    if found:

        notes.append(
            f"For your privacy, I removed the {', '.join(found)} "
            "from your message before sending it to the AI agents. "
            "Never share full card numbers or passwords in chat."
        )

    return GuardrailResult(
        allowed=True,
        text=redacted,
        notes=notes
    )


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def check_output(client: OpenAI, answer: str) -> GuardrailResult:

    lowered = answer.lower()

    if any(marker in lowered for marker in SYSTEM_PROMPT_MARKERS):

        return GuardrailResult(
            allowed=False,
            text=answer,
            reason=(
                "Sorry, I couldn't produce a safe answer to that. "
                "Please try rephrasing your question."
            ),
            notes=["Blocked: response exposed internal instructions."]
        )

    flagged = moderate(client, answer)

    if flagged:

        return GuardrailResult(
            allowed=False,
            text=answer,
            reason=(
                "Sorry, I couldn't produce a safe answer to that. "
                "Please try rephrasing your question."
            ),
            notes=[f"Flagged: {', '.join(flagged)}"]
        )

    redacted, found = redact_pii(answer, OUTPUT_REDACT)

    notes = []

    if found:
        notes.append(f"Removed {', '.join(found)} from the response.")

    return GuardrailResult(
        allowed=True,
        text=redacted,
        notes=notes
    )



# ============================================================
# AGENT SETTINGS
# ============================================================

DEFAULT_MODEL = "gpt-4o-mini"

# Where save_to_file writes. Evaluation runs point this elsewhere
# so they don't overwrite the chat's answers.txt.
ANSWERS_FILE = "answers.txt"


# ============================================================
# TOOL - SAVE RESULTS
# ============================================================

def save_to_file(
    query: str,
    answer1: str,
    answer2: str
) -> str:

    with open(
        ANSWERS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Customer Support Query\n"
        )

        file.write(
            "======================\n\n"
        )

        file.write(
            f"Query:\n{query}\n\n"
        )

        file.write(
            f"Assistant Answer:\n{answer1}\n\n"
        )

        file.write(
            f"Web Search Answer:\n{answer2}\n"
        )

    return "File saved successfully."


# ============================================================
# TOOL - WEB SEARCH
# ============================================================

def web_search_tool(query: str) -> str:

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": 1
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=20
        )

        if response.status_code != 200:

            return (
                f"Web search error: "
                f"{response.status_code}"
            )

        data = response.json()

        if not data.get("organic"):

            return "No web search results found."

        result = data["organic"][0]

        title = result.get(
            "title",
            ""
        )

        snippet = result.get(
            "snippet",
            ""
        )

        link = result.get(
            "link",
            ""
        )

        return (
            f"Title: {title}\n"
            f"Snippet: {snippet}\n"
            f"Source: {link}"
        )

    except Exception as e:

        return (
            f"Web search failed: {str(e)}"
        )


# ============================================================
# AUTOGEN WORKFLOW
# ============================================================

async def run_customer_support(query: str, model: str = DEFAULT_MODEL):

    # --------------------------------------------------------
    # MODEL CLIENT
    # --------------------------------------------------------

    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key
    )


    # --------------------------------------------------------
    # AGENT 1 - ASSISTANT
    # --------------------------------------------------------

    assistant = AssistantAgent(

        name="Assistant",

        model_client=model_client,

        system_message="""
        You are the first customer support assistant.

        Answer the user's question using your own knowledge.

        Do not use any tools.

        Provide a clear and useful answer.
        """
    )


    # --------------------------------------------------------
    # AGENT 2 - WEB SEARCH ASSISTANT
    # --------------------------------------------------------

    web_agent = AssistantAgent(

        name="WebSearchAssistant",

        model_client=model_client,

        system_message="""
        You are the web search customer support assistant.

        Use the web search tool to find relevant information
        for the user's question.

        Analyze the search result and provide a useful answer.

        Do not merely say that you searched the web.
        Provide the actual answer.
        """,

        tools=[
            web_search_tool
        ]
    )


    # --------------------------------------------------------
    # AGENT 3 - ENTRY / RECONCILIATION AGENT
    # --------------------------------------------------------

    entry_agent = AssistantAgent(

        name="EntryAgent",

        model_client=model_client,

        system_message="""
        You are the final customer support and reconciliation agent.

        You receive:

        1. The original user query.
        2. The answer from Assistant.
        3. The answer from WebSearchAssistant.

        STEP 1:
        Extract the original query and both answers.

        STEP 2:
        Call save_to_file with:

        - query
        - answer1
        - answer2

        STEP 3:
        After the file is successfully saved, reconcile
        the two answers.

        Produce one final customer-friendly answer.

        The final answer should:

        - directly answer the user's question
        - combine useful information from both agents
        - remove unnecessary duplication
        - resolve obvious inconsistencies when possible
        - be clear and concise

        IMPORTANT:

        Save the file BEFORE generating the final answer.

        After producing the final answer, write TERMINATE
        as the very last word.
        """,

        tools=[
            save_to_file
        ],

        # Save and answer in the same turn. Without this the agent's
        # turn ends after the tool call, and round-robin re-runs the
        # Assistant and web search before the final answer.

        reflect_on_tool_use=True
    )


    # --------------------------------------------------------
    # TERMINATION
    # --------------------------------------------------------

    # The message cap is a safety net: if the Entry Agent never says
    # TERMINATE, round-robin would otherwise loop (and bill) forever.
    # A normal run is 4 chat messages: user task, Assistant,
    # WebSearchAssistant, EntryAgent. The cap counts the user task.

    termination_condition = (
        TextMentionTermination("TERMINATE")
        | MaxMessageTermination(5)
    )


    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    team = RoundRobinGroupChat(

        participants=[
            assistant,
            web_agent,
            entry_agent
        ],

        termination_condition=termination_condition
    )


    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    try:

        result = await team.run(
            task=query
        )

    finally:

        await model_client.close()

    return result


# ============================================================
# EXTRACT RESPONSES
# ============================================================

def extract_responses(result) -> dict:

    responses = {
        "assistant_answer": None,
        "web_answer": None,
        "final_answer": None,
        "save_status": None
    }

    for message in result.messages:

        source = getattr(message, "source", None)

        message_type = type(message).__name__

        if source == "Assistant" and message_type == "TextMessage":

            responses["assistant_answer"] = message.content

        elif (
            source == "WebSearchAssistant"
            and message_type == "ToolCallSummaryMessage"
        ):

            responses["web_answer"] = message.content

        elif (
            source == "EntryAgent"
            and message_type == "ToolCallExecutionEvent"
        ):

            # With reflect_on_tool_use the save shows up as an
            # execution event rather than a summary message.

            responses["save_status"] = (
                message.content[0].content if message.content else None
            )

        elif source == "EntryAgent" and message_type == "TextMessage":

            # Skip a bare "TERMINATE" so it can't blank out an
            # answer the agent already gave.

            answer = message.content.replace("TERMINATE", "").strip()

            if answer:
                responses["final_answer"] = answer

    return responses


# ============================================================
# EVALUATION - CONFIGURATION
# ============================================================

DEFAULT_JUDGE_MODEL = "gpt-4o"

# USD per 1M tokens (input, output). Check these against current
# OpenAI pricing before relying on the cost numbers.

MODEL_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

# USD per Serper search. Depends on your plan's credit pack.
SEARCH_PRICE = 0.001

AGENTS = ["Assistant", "WebSearchAssistant", "EntryAgent"]

# Test questions with the key points a good answer should cover.

EVAL_DATASET = [
    {
        "id": "password-reset",
        "category": "account",
        "question": "How do I reset my password if I no longer have access to my email?",
        "expected_points": [
            "Use the account recovery / forgot password flow",
            "Verify identity another way (phone number, security questions or ID)",
            "Contact customer support if self-service recovery fails",
            "Update the email address on the account once access is restored",
        ]
    },
    {
        "id": "refund-policy",
        "category": "billing",
        "question": "What is the typical refund policy for online purchases?",
        "expected_points": [
            "Refund windows are commonly around 14-30 days",
            "Items usually must be unused / in original condition with proof of purchase",
            "Refunds typically go back to the original payment method",
            "Policies vary by retailer, so check the specific store's terms",
        ]
    },
    {
        "id": "track-international",
        "category": "shipping",
        "question": "How can I track an international shipment?",
        "expected_points": [
            "Use the tracking number from the shipping confirmation",
            "Track on the carrier's website or app",
            "Tracking may switch to the destination country's postal service",
            "Customs clearance can pause tracking updates",
        ]
    },
    {
        "id": "damaged-order",
        "category": "orders",
        "question": "My order arrived damaged. What should I do?",
        "expected_points": [
            "Take photos of the damaged item and packaging",
            "Contact the seller or customer support promptly",
            "Keep the original packaging",
            "Ask for a replacement or refund",
        ]
    },
    {
        "id": "cancel-subscription",
        "category": "billing",
        "question": "How do I cancel a subscription and avoid being charged again?",
        "expected_points": [
            "Cancel through account settings or the platform where you subscribed (e.g. app store)",
            "Cancel before the next billing date",
            "Keep the cancellation confirmation",
            "Check your statement and contact support about unexpected charges",
        ]
    },
    {
        "id": "double-charge",
        "category": "billing",
        "question": "I was charged twice for the same order. How do I get my money back?",
        "expected_points": [
            "Check whether one charge is a pending authorization that will drop off",
            "Contact the merchant with order details and statement proof",
            "Request a refund of the duplicate charge",
            "Dispute with the bank or card issuer if the merchant does not resolve it",
        ]
    },
    {
        "id": "change-address",
        "category": "orders",
        "question": "Can I change the shipping address after placing an order?",
        "expected_points": [
            "Possible only before the order ships",
            "Contact customer support or edit the order quickly",
            "After shipping, the carrier may offer a redirect / delivery change",
            "Otherwise cancel and reorder, or return the item",
        ]
    },
    {
        "id": "gdpr-delete",
        "category": "privacy",
        "question": "How do I request that a company delete my personal data?",
        "expected_points": [
            "Submit a data deletion request (right to erasure) to the company",
            "Use the privacy settings, privacy page or data protection contact",
            "Company may need to verify your identity",
            "Companies generally must respond within a legal deadline (e.g. one month under GDPR)",
        ]
    },
    {
        "id": "package-stolen",
        "category": "shipping",
        "question": "My package says delivered but I never received it. What can I do?",
        "expected_points": [
            "Check around the delivery location and with neighbours",
            "Wait a day, as carriers sometimes mark delivered early",
            "Contact the carrier and the seller",
            "File a claim / request a replacement or refund",
        ]
    },
    {
        "id": "warranty-claim",
        "category": "product",
        "question": "How do I make a warranty claim for a faulty product?",
        "expected_points": [
            "Check the warranty terms and whether it is still valid",
            "Have proof of purchase ready",
            "Contact the manufacturer or retailer with a description of the fault",
            "Expect repair, replacement or refund depending on the terms",
        ]
    },
]



# ============================================================
# EVALUATION - JUDGE PROMPTS
# ============================================================

RETRIEVAL_JUDGE_PROMPT = """
You are evaluating the web retrieval step of a customer support system.

User question:
{question}

Search query the agent sent to the search engine:
{search_query}

Search result returned (title, snippet, source):
{search_result}

Expected key points a good answer should contain:
{expected_points}

Score each from 1 (very poor) to 5 (excellent):
- query_relevance: does the search query capture the user's intent?
- result_relevance: is the returned result on-topic and useful for
  answering the question?
- source_quality: is the source trustworthy/authoritative for this
  topic (official docs, reputable sites > forums/spam)?

Also list the numbers of the expected key points that are supported
by the search result.

Return JSON only:
{{"query_relevance": int, "result_relevance": int, "source_quality": int,
  "points_supported": [int], "comment": str}}
"""

ANSWER_JUDGE_PROMPT = """
You are evaluating the final answer of a customer support system. The
final answer was written by a reconciliation agent that merged two
inputs: a knowledge-based answer and a web search result.

User question:
{question}

Expected key points:
{expected_points}

Input 1 - knowledge assistant answer:
{assistant_answer}

Input 2 - web search result:
{web_answer}

FINAL ANSWER (the one being evaluated):
{final_answer}

Score the FINAL ANSWER from 1 (very poor) to 5 (excellent):
- correctness: factually accurate and actually answers the question.
- faithfulness: claims are supported by the inputs or well-established
  common knowledge; no fabricated policies, numbers or links.
- reconciliation: combines the useful parts of BOTH inputs, removes
  duplication, and resolves conflicts sensibly.
- clarity: clear, well structured, concise, friendly support tone.

List the numbers of the expected key points covered by the FINAL
ANSWER, and separately those covered by Input 1 alone (to measure
what the reconciliation step adds).

Set "pass" to true only if the final answer would fully satisfy a
real customer.

Return JSON only:
{{"correctness": int, "faithfulness": int, "reconciliation": int,
  "clarity": int, "points_covered_final": [int],
  "points_covered_assistant": [int], "pass": bool, "comment": str}}
"""


# ============================================================
# EVALUATION - HELPERS
# ============================================================

def token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:

    if model not in MODEL_PRICES:

        raise ValueError(
            f"No price for model '{model}'. Add it to MODEL_PRICES."
        )

    input_price, output_price = MODEL_PRICES[model]

    return (
        prompt_tokens * input_price
        + completion_tokens * output_price
    ) / 1_000_000


def percentile(values: list[float], pct: float) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)

    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))

    return ordered[index]


def mean(values: list[float]) -> float:

    values = [v for v in values if v is not None]

    return statistics.mean(values) if values else 0.0


def numbered(points: list[str]) -> str:

    return "\n".join(f"{i}. {point}" for i, point in enumerate(points, 1))


def coverage(covered: list, expected: list[str]) -> float:

    # The judge returns point numbers (1-based); ignore duplicates and
    # anything out of range.

    hits = {
        int(n) for n in covered
        if str(n).isdigit() and 1 <= int(n) <= len(expected)
    }

    return len(hits) / len(expected) if expected else 0.0


class Judge:

    def __init__(self, client: OpenAI, model: str):

        self.client = client

        self.model = model

        self.prompt_tokens = 0

        self.completion_tokens = 0

    def __call__(self, prompt: str) -> dict:

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )

        self.prompt_tokens += response.usage.prompt_tokens

        self.completion_tokens += response.usage.completion_tokens

        return json.loads(response.choices[0].message.content)

    @property
    def cost(self) -> float:

        return token_cost(self.model, self.prompt_tokens, self.completion_tokens)


# ============================================================
# EVALUATION - TRACE ANALYSIS (cost + latency + retrieval facts)
# ============================================================

def analyse_trace(result, start: datetime) -> dict:

    messages = sorted(result.messages, key=lambda m: m.created_at)

    usage = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    agent_seconds = defaultdict(float)

    searches = []

    pending_search = None

    previous_time = start

    for message in messages:

        source = message.source

        message_type = type(message).__name__

        # Latency: time since the previous message is work done by
        # this message's author.

        if source in AGENTS:

            agent_seconds[source] += (
                message.created_at - previous_time
            ).total_seconds()

        previous_time = max(previous_time, message.created_at)

        # Cost: AutoGen attaches token usage to the model-call events.

        if message.models_usage:

            usage[source]["prompt_tokens"] += message.models_usage.prompt_tokens

            usage[source]["completion_tokens"] += message.models_usage.completion_tokens

        # Retrieval: pair web_search_tool requests with their results.

        if source == "WebSearchAssistant":

            if message_type == "ToolCallRequestEvent":

                for call in message.content:

                    if call.name == "web_search_tool":

                        pending_search = {
                            "query": json.loads(call.arguments).get("query", ""),
                            "requested_at": message.created_at
                        }

            elif message_type == "ToolCallExecutionEvent" and pending_search:

                output = message.content[0].content if message.content else ""

                searches.append(
                    {
                        "query": pending_search["query"],
                        "result": output,
                        "seconds": (
                            message.created_at - pending_search["requested_at"]
                        ).total_seconds(),
                        "error": output.startswith(
                            ("Web search error", "Web search failed", "No web search results")
                        )
                    }
                )

                pending_search = None

    # Process checks for the Entry Agent

    entry_types = [
        type(m).__name__ for m in messages if m.source == "EntryAgent"
    ]

    saved_before_answer = (
        "ToolCallExecutionEvent" in entry_types
        and "TextMessage" in entry_types
        and entry_types.index("ToolCallExecutionEvent")
        < entry_types.index("TextMessage")
    )

    return {
        "usage": {agent: dict(usage[agent]) for agent in AGENTS},
        "agent_seconds": {agent: round(agent_seconds[agent], 3) for agent in AGENTS},
        "searches": searches,
        "saved_before_answer": saved_before_answer,
        "terminated_cleanly": "TERMINATE" in (result.stop_reason or ""),
        "message_count": len(messages)
    }


# ============================================================
# EVALUATION - EVALUATE ONE QUESTION
# ============================================================

def evaluate_case(case: dict, client: OpenAI, judge: Judge, model: str) -> dict:

    record = {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"]
    }

    expected = case["expected_points"]

    total_start = time.perf_counter()

    # ---------- Input guardrail ----------

    t0 = time.perf_counter()

    input_check = check_input(client, case["question"])

    record["latency_input_guardrail"] = time.perf_counter() - t0

    if not input_check.allowed:

        record["error"] = f"Blocked by input guardrail: {input_check.reason}"

        return record

    # ---------- Agents ----------

    agent_start = datetime.now().astimezone()

    t0 = time.perf_counter()

    try:

        result = asyncio.run(run_customer_support(input_check.text, model=model))

    except Exception as e:

        record["error"] = f"Pipeline failed: {e}"

        return record

    record["latency_agents"] = time.perf_counter() - t0

    responses = extract_responses(result)

    trace = analyse_trace(result, agent_start)

    # ---------- Output guardrail ----------

    final_answer = responses["final_answer"]

    t0 = time.perf_counter()

    output_check = check_output(client, final_answer) if final_answer else None

    record["latency_output_guardrail"] = time.perf_counter() - t0

    record["latency_total"] = time.perf_counter() - total_start

    record["output_blocked"] = bool(output_check and not output_check.allowed)

    # ---------- Latency detail ----------

    for agent, seconds in trace["agent_seconds"].items():

        record[f"latency_{agent}"] = seconds

    record["latency_search_api"] = sum(s["seconds"] for s in trace["searches"])

    # ---------- Cost ----------

    total_tokens_cost = 0.0

    for agent, usage in trace["usage"].items():

        record[f"tokens_{agent}"] = usage["prompt_tokens"] + usage["completion_tokens"]

        total_tokens_cost += token_cost(
            model, usage["prompt_tokens"], usage["completion_tokens"]
        )

    record["cost_llm"] = total_tokens_cost

    record["cost_search"] = len(trace["searches"]) * SEARCH_PRICE

    record["cost_total"] = record["cost_llm"] + record["cost_search"]

    # ---------- Retrieval quality ----------

    search = trace["searches"][-1] if trace["searches"] else None

    record["search_called"] = search is not None

    record["search_count"] = len(trace["searches"])

    record["search_error"] = bool(search and search["error"])

    record["search_query"] = search["query"] if search else None

    record["search_result"] = search["result"] if search else None

    if search and not search["error"]:

        retrieval = judge(
            RETRIEVAL_JUDGE_PROMPT.format(
                question=case["question"],
                search_query=search["query"],
                search_result=search["result"],
                expected_points=numbered(expected)
            )
        )

        record["retrieval_query_relevance"] = retrieval["query_relevance"]

        record["retrieval_result_relevance"] = retrieval["result_relevance"]

        record["retrieval_source_quality"] = retrieval["source_quality"]

        record["retrieval_context_recall"] = coverage(
            retrieval["points_supported"], expected
        )

        record["retrieval_comment"] = retrieval.get("comment")

    # ---------- Answer quality (Entry Agent) ----------

    record["saved_before_answer"] = trace["saved_before_answer"]

    record["terminated_cleanly"] = trace["terminated_cleanly"]

    record["assistant_answer"] = responses["assistant_answer"]

    record["final_answer"] = final_answer

    if final_answer:

        answer = judge(
            ANSWER_JUDGE_PROMPT.format(
                question=case["question"],
                expected_points=numbered(expected),
                assistant_answer=responses["assistant_answer"] or "(none)",
                web_answer=responses["web_answer"] or "(none)",
                final_answer=final_answer
            )
        )

        record["answer_correctness"] = answer["correctness"]

        record["answer_faithfulness"] = answer["faithfulness"]

        record["answer_reconciliation"] = answer["reconciliation"]

        record["answer_clarity"] = answer["clarity"]

        record["answer_pass"] = bool(answer["pass"])

        record["answer_recall_final"] = coverage(answer["points_covered_final"], expected)

        record["answer_recall_assistant_only"] = coverage(
            answer["points_covered_assistant"], expected
        )

        record["answer_comment"] = answer.get("comment")

    else:

        record["error"] = "Entry Agent returned no final answer"

    return record


# ============================================================
# EVALUATION - SUMMARY
# ============================================================

def summarise(records: list[dict], judge: Judge, model: str) -> dict:

    ok = [r for r in records if "error" not in r]

    def col(name):
        return [r[name] for r in ok if r.get(name) is not None]

    def rate(name):
        values = col(name)
        return sum(bool(v) for v in values) / len(values) if values else 0.0

    totals = col("latency_total")

    cost_per_query = mean(col("cost_total"))

    return {
        "run": {
            "model": model,
            "judge_model": judge.model,
            "questions": len(records),
            "succeeded": len(ok),
            "failed": [
                {"id": r["id"], "error": r["error"]}
                for r in records if "error" in r
            ]
        },
        "retrieval": {
            "search_call_rate": rate("search_called"),
            "search_error_rate": rate("search_error"),
            "avg_query_relevance_1to5": mean(col("retrieval_query_relevance")),
            "avg_result_relevance_1to5": mean(col("retrieval_result_relevance")),
            "avg_source_quality_1to5": mean(col("retrieval_source_quality")),
            "avg_context_recall": mean(col("retrieval_context_recall"))
        },
        "answer_quality": {
            "pass_rate": rate("answer_pass"),
            "avg_correctness_1to5": mean(col("answer_correctness")),
            "avg_faithfulness_1to5": mean(col("answer_faithfulness")),
            "avg_reconciliation_1to5": mean(col("answer_reconciliation")),
            "avg_clarity_1to5": mean(col("answer_clarity")),
            "avg_key_point_recall_final": mean(col("answer_recall_final")),
            "avg_key_point_recall_assistant_only": mean(
                col("answer_recall_assistant_only")
            ),
            "saved_before_answer_rate": rate("saved_before_answer"),
            "terminated_cleanly_rate": rate("terminated_cleanly"),
            "output_blocked_rate": rate("output_blocked")
        },
        "cost_usd": {
            "avg_per_query": cost_per_query,
            "avg_llm_per_query": mean(col("cost_llm")),
            "avg_search_per_query": mean(col("cost_search")),
            "projected_per_1000_queries": cost_per_query * 1000,
            "avg_tokens_per_agent": {
                agent: mean(col(f"tokens_{agent}")) for agent in AGENTS
            },
            "total_run_pipeline": sum(col("cost_total")),
            "total_run_judge": judge.cost
        },
        "latency_seconds": {
            "p50_total": percentile(totals, 50),
            "p95_total": percentile(totals, 95),
            "max_total": max(totals, default=0.0),
            "avg_input_guardrail": mean(col("latency_input_guardrail")),
            "avg_agents": mean(col("latency_agents")),
            "avg_output_guardrail": mean(col("latency_output_guardrail")),
            "avg_per_agent": {
                agent: mean(col(f"latency_{agent}")) for agent in AGENTS
            },
            "avg_search_api": mean(col("latency_search_api"))
        }
    }


# ============================================================
# UI CONSTANTS
# ============================================================

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🤖"

SUGGESTED_QUESTIONS = [
    "How do I reset my password if I no longer have access to my email?",
    "What is the typical refund policy for online purchases?",
    "How can I track an international shipment?",
    "My order arrived damaged. What should I do?",
]


# ============================================================
# UI HELPERS
# ============================================================

def render_assistant_message(message: dict, index: int):

    # Agent details (only present for multi-agent answers)

    details = message.get("details") or {}

    if details.get("assistant_answer"):

        with st.expander("🧠 Knowledge Assistant answer"):

            st.markdown(details["assistant_answer"])

    if details.get("web_answer"):

        with st.expander("🌐 Web search findings"):

            st.markdown(details["web_answer"])

    # Plain replies (greetings, fallbacks) need no card

    if not details:

        st.markdown(message["content"])

        render_guard_notes(message)

        return

    with st.container(key=f"answer-{index}"):

        st.html(
            '<span class="answer-badge">💬 Final support answer</span>'
        )

        st.markdown(message["content"])

        if details.get("saved"):

            st.html(
                '<div class="saved-note">✓ Query and agent responses '
                'saved to answers.txt</div>'
            )

        render_guard_notes(message)


def render_guard_notes(message: dict):

    for note in message.get("notes") or []:

        st.html(
            f'<div class="guard-note">🛡️ {html.escape(note)}</div>'
        )


def append_message(
    role: str,
    content: str,
    details: dict | None = None,
    notes: list[str] | None = None
):

    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "details": details,
            "notes": notes
        }
    )


def reply(content: str, notes: list[str] | None = None):

    append_message("assistant", content, notes=notes)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

        render_assistant_message(
            st.session_state.messages[-1],
            len(st.session_state.messages) - 1
        )


# ============================================================
# SIDEBAR
# ============================================================

CHAT_MODE = "💬 Chat"
EVAL_MODE = "📊 Evaluation"

with st.sidebar:

    mode = st.radio(
        "Mode",
        [CHAT_MODE, EVAL_MODE],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.html(
        """
<div class="side-brand">🤖 Support Copilot</div>
<div class="side-muted">Powered by AutoGen · gpt-4o-mini</div>

<div class="section-label">How it works</div>

<div class="pipeline-step">
  <div class="step-icon tint-indigo">🧠</div>
  <div>
    <div class="step-title">Knowledge Assistant</div>
    <div class="step-text">Drafts an answer from the model's own knowledge.</div>
  </div>
</div>

<div class="pipeline-step">
  <div class="step-icon tint-sky">🌐</div>
  <div>
    <div class="step-title">Web Search Assistant</div>
    <div class="step-text">Looks up current information on the web.</div>
  </div>
</div>

<div class="pipeline-step">
  <div class="step-icon tint-pink">📝</div>
  <div>
    <div class="step-title">Reconciliation Entry Agent</div>
    <div class="step-text">Saves both answers and merges them into one reply.</div>
  </div>
</div>

<div class="section-label" style="margin-top:18px">Guardrails</div>

<div class="pipeline-step">
  <div class="step-icon tint-amber">🛡️</div>
  <div>
    <div class="step-title">Input checks</div>
    <div class="step-text">Blocks prompt injection and harmful content; removes card numbers, passwords and other personal data.</div>
  </div>
</div>

<div class="pipeline-step">
  <div class="step-icon tint-amber">✅</div>
  <div>
    <div class="step-title">Output checks</div>
    <div class="step-text">Moderates every answer and blocks leaked secrets or internal instructions.</div>
  </div>
</div>
"""
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        width="stretch"
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# EVALUATION PAGE
# ============================================================

def run_evaluation(cases: list[dict], model: str, judge_model: str) -> dict:

    global ANSWERS_FILE

    # Keep eval runs from overwriting the chat's answers.txt.
    # Reset automatically on the next Streamlit rerun.

    ANSWERS_FILE = "eval_answers.txt"

    judge = Judge(openai_client, judge_model)

    records = []

    progress = st.progress(0.0)

    for i, case in enumerate(cases):

        progress.progress(
            i / len(cases),
            text=f"Evaluating {i + 1}/{len(cases)}: {case['question']}"
        )

        records.append(
            evaluate_case(case, openai_client, judge, model)
        )

    progress.empty()

    return {
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "records": records,
        "summary": summarise(records, judge, model)
    }


def score(value: float) -> str:

    return f"{value:.2f} / 5"


def pct(value: float) -> str:

    return f"{value:.0%}"


def usd(value: float) -> str:

    return f"${value:.4f}" if value < 1 else f"${value:,.2f}"


def secs(value: float) -> str:

    return f"{value:.1f}s"


def metric_row(items: list[tuple[str, str, str | None]]):

    for column, (label, value, help_text) in zip(st.columns(len(items)), items):

        column.metric(label, value, help=help_text, border=True)


def render_evaluation_page():

    st.html(
        """
<div class="hero">
  <div class="hero-eyebrow">Quality · Cost · Speed</div>
  <div class="hero-title">📊 Pipeline Evaluation</div>
  <div class="hero-subtitle">
    Runs test questions through the same guardrails and agents as the
    chat, then scores retrieval quality, the Entry Agent's final answer,
    cost and latency. Answer and retrieval scores come from an LLM judge.
  </div>
</div>
"""
    )

    # ---------- Controls ----------

    with st.container(border=True):

        col1, col2, col3 = st.columns(3)

        count = col1.number_input(
            "Questions to run",
            min_value=1,
            max_value=len(EVAL_DATASET),
            value=len(EVAL_DATASET)
        )

        models = list(MODEL_PRICES)

        model = col2.selectbox(
            "Agent model",
            models,
            index=models.index(DEFAULT_MODEL)
        )

        judge_model = col3.selectbox(
            "Judge model",
            models,
            index=models.index(DEFAULT_JUDGE_MODEL),
            help="Use a stronger model than the agents for fairer grading."
        )

        st.caption(
            "Each question makes real OpenAI and Serper calls and takes "
            "roughly 10–30 seconds. Prices come from MODEL_PRICES and "
            "SEARCH_PRICE in app.py."
        )

        run = st.button("▶ Run evaluation", type="primary", width="stretch")

    with st.expander(f"Test questions ({len(EVAL_DATASET)})"):

        st.dataframe(
            [
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "expected points": " • ".join(case["expected_points"])
                }
                for case in EVAL_DATASET
            ],
            hide_index=True
        )

    if run:

        st.session_state.eval_run = run_evaluation(
            EVAL_DATASET[:count],
            model,
            judge_model
        )

    data = st.session_state.get("eval_run")

    if not data:

        st.info("Run an evaluation to see results here.")

        return

    summary = data["summary"]

    records = pd.DataFrame(data["records"])

    info = summary["run"]

    st.caption(
        f"Last run {data['finished_at']} · agents: {info['model']} · "
        f"judge: {info['judge_model']} · "
        f"{info['succeeded']}/{info['questions']} questions succeeded"
    )

    for failure in info["failed"]:

        st.warning(f"**{failure['id']}**: {failure['error']}")

    ok = records[records["error"].isna()] if "error" in records else records

    retrieval_tab, answer_tab, cost_tab, latency_tab, detail_tab = st.tabs(
        ["🔎 Retrieval", "✅ Answer quality", "💰 Cost", "⏱️ Latency", "📋 Per question"]
    )

    # ---------- 1. Retrieval ----------

    with retrieval_tab:

        r = summary["retrieval"]

        metric_row([
            ("Search call rate", pct(r["search_call_rate"]),
             "Share of questions where the Web Search Assistant actually searched."),
            ("Search error rate", pct(r["search_error_rate"]),
             "API errors or empty results."),
            ("Context recall", pct(r["avg_context_recall"]),
             "Share of expected key points supported by the search result."),
        ])

        metric_row([
            ("Query relevance", score(r["avg_query_relevance_1to5"]),
             "Does the search query capture the user's intent?"),
            ("Result relevance", score(r["avg_result_relevance_1to5"]),
             "Is the returned result on-topic and useful?"),
            ("Source quality", score(r["avg_source_quality_1to5"]),
             "Is the source trustworthy for this topic?"),
        ])

        columns = [
            c for c in [
                "id", "search_query", "retrieval_result_relevance",
                "retrieval_context_recall", "retrieval_comment", "search_result"
            ] if c in ok
        ]

        st.dataframe(
            ok[columns],
            hide_index=True,
            column_config={
                "retrieval_result_relevance": st.column_config.NumberColumn("relevance", format="%d / 5"),
                "retrieval_context_recall": st.column_config.ProgressColumn("context recall", min_value=0, max_value=1, format="percent"),
            }
        )

    # ---------- 2. Answer quality ----------

    with answer_tab:

        a = summary["answer_quality"]

        gain = a["avg_key_point_recall_final"] - a["avg_key_point_recall_assistant_only"]

        metric_row([
            ("Pass rate", pct(a["pass_rate"]),
             "Judge says the answer would fully satisfy a customer."),
            ("Key-point recall", pct(a["avg_key_point_recall_final"]),
             "Share of expected key points in the final answer."),
            ("Gain from reconciliation", f"{gain:+.0%}",
             "Final answer recall minus the Knowledge Assistant's alone."),
        ])

        metric_row([
            ("Correctness", score(a["avg_correctness_1to5"]), None),
            ("Faithfulness", score(a["avg_faithfulness_1to5"]),
             "No fabricated policies, numbers or links."),
            ("Reconciliation", score(a["avg_reconciliation_1to5"]),
             "Merges both inputs and resolves conflicts."),
            ("Clarity", score(a["avg_clarity_1to5"]), None),
        ])

        metric_row([
            ("Saved before answering", pct(a["saved_before_answer_rate"]),
             "Entry Agent called save_to_file before its final answer."),
            ("Terminated cleanly", pct(a["terminated_cleanly_rate"]),
             "Run ended with TERMINATE rather than the message cap."),
            ("Blocked by output guardrail", pct(a["output_blocked_rate"]), None),
        ])

        columns = [
            c for c in [
                "id", "answer_pass", "answer_correctness", "answer_faithfulness",
                "answer_reconciliation", "answer_clarity", "answer_recall_final",
                "answer_comment", "final_answer"
            ] if c in ok
        ]

        st.dataframe(
            ok[columns],
            hide_index=True,
            column_config={
                "answer_recall_final": st.column_config.ProgressColumn("recall", min_value=0, max_value=1, format="percent"),
            }
        )

    # ---------- 3. Cost ----------

    with cost_tab:

        c = summary["cost_usd"]

        metric_row([
            ("Avg cost / question", usd(c["avg_per_query"]), None),
            ("Per 1,000 questions", usd(c["projected_per_1000_queries"]), None),
            ("LLM / search split",
             f"{usd(c['avg_llm_per_query'])} / {usd(c['avg_search_per_query'])}", None),
        ])

        metric_row([
            ("This run (pipeline)", usd(c["total_run_pipeline"]), None),
            ("This run (judge)", usd(c["total_run_judge"]),
             "Evaluation overhead, not part of production cost."),
        ])

        tokens = c["avg_tokens_per_agent"]

        total_tokens = sum(tokens.values()) or 1

        st.dataframe(
            [
                {"agent": agent, "avg tokens": round(value), "share": value / total_tokens}
                for agent, value in tokens.items()
            ],
            hide_index=True,
            column_config={
                "share": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
            }
        )

    # ---------- 4. Latency ----------

    with latency_tab:

        l = summary["latency_seconds"]

        metric_row([
            ("p50 end-to-end", secs(l["p50_total"]), None),
            ("p95 end-to-end", secs(l["p95_total"]), None),
            ("Slowest", secs(l["max_total"]), None),
        ])

        stages = {
            "Input guardrail": l["avg_input_guardrail"],
            **{f"Agent: {agent}": value for agent, value in l["avg_per_agent"].items()},
            "Output guardrail": l["avg_output_guardrail"],
        }

        total_stage = sum(stages.values()) or 1

        st.dataframe(
            [
                {"stage": stage, "avg seconds": round(value, 2), "share": value / total_stage}
                for stage, value in stages.items()
            ],
            hide_index=True,
            column_config={
                "share": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
            }
        )

        st.caption(
            f"Web Search Assistant time includes {secs(l['avg_search_api'])} "
            "on average waiting for the Serper API."
        )

    # ---------- Per question + downloads ----------

    with detail_tab:

        st.dataframe(records, hide_index=True)

        col1, col2 = st.columns(2)

        col1.download_button(
            "⬇ Download results (CSV)",
            records.to_csv(index=False),
            file_name="eval_results.csv",
            mime="text/csv",
            width="stretch"
        )

        col2.download_button(
            "⬇ Download summary (JSON)",
            json.dumps({"summary": summary, "records": data["records"]}, indent=2),
            file_name="eval_summary.json",
            mime="application/json",
            width="stretch"
        )


if mode == EVAL_MODE:

    render_evaluation_page()

    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.html(
    """
<div class="hero">
  <div class="hero-eyebrow"><span class="live-dot"></span>Online · 3 agents ready</div>
  <div class="hero-title">AI Customer Support</div>
  <div class="hero-subtitle">
    Multi-agent assistance that combines built-in knowledge, live web search
    and intelligent reconciliation into one clear answer.
  </div>
  <div class="agent-chips">
    <span class="agent-chip">🧠 Knowledge</span>
    <span class="agent-chip">🌐 Web search</span>
    <span class="agent-chip">📝 Reconciliation Entry Agent</span>
  </div>
</div>
"""
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.html(
        """
<div class="welcome">
  <div class="welcome-title">👋 How can we help today?</div>
  <div class="welcome-text">Ask a support question and I'll coordinate multiple AI agents to answer it.</div>
</div>

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon tint-indigo">🧠</div>
    <div class="feature-title">Expert knowledge</div>
    <div class="feature-text">Instant answers drawn from broad product and support know-how.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon tint-sky">🌐</div>
    <div class="feature-title">Live research</div>
    <div class="feature-text">Fresh information from the web to keep answers up to date.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon tint-pink">✨</div>
    <div class="feature-title">One clear reply</div>
    <div class="feature-text">Both sources reconciled into a single, friendly response.</div>
  </div>
</div>

<div class="section-label">Try asking</div>
"""
    )

    columns = st.columns(2)

    for i, question in enumerate(SUGGESTED_QUESTIONS):

        with columns[i % 2]:

            if st.button(question, key=f"suggest-{i}", width="stretch"):

                st.session_state.pending_query = question

                st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for index, message in enumerate(st.session_state.messages):

    if message["role"] == "user":

        with st.chat_message("user", avatar=USER_AVATAR):

            st.markdown(message["content"])

    else:

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

            render_assistant_message(message, index)


# ============================================================
# BOTTOM INPUT
# ============================================================

query = st.chat_input(
    "Ask your customer support question..."
)

if not query:

    query = st.session_state.pop("pending_query", None)


# ============================================================
# PROCESS QUERY
# ============================================================

if query:

    # ========================================================
    # INPUT GUARDRAIL
    # ========================================================
    # Runs before anything is shown or stored, so redacted data
    # (card numbers, passwords, ...) never lands in the chat
    # history, the agents, the web search API or answers.txt.

    input_check = check_input(
        openai_client,
        query
    )

    if input_check.allowed:

        query = input_check.text


    # ========================================================
    # SHOW USER MESSAGE
    # ========================================================

    with st.chat_message("user", avatar=USER_AVATAR):

        st.markdown(query)

    append_message("user", query)


    # ========================================================
    # BLOCKED BY INPUT GUARDRAIL
    # ========================================================

    if not input_check.allowed:

        reply(
            input_check.reason,
            notes=["Blocked by input guardrail."] + input_check.notes
        )

        st.stop()


    # ========================================================
    # CHECK GREETING
    # ========================================================

    greeting_response = detect_greeting(
        query
    )


    # ========================================================
    # GREETING
    # ========================================================

    if greeting_response:

        reply(
            greeting_response,
            notes=input_check.notes
        )


    # ========================================================
    # CUSTOMER SUPPORT QUESTION
    # ========================================================

    else:

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

            # ------------------------------------------------
            # AGENT EXECUTION STATUS
            # ------------------------------------------------

            with st.status(
                "AI agents are working on your question...",
                expanded=True
            ) as status:

                st.write(
                    "🧠 Knowledge Assistant is analyzing the question"
                )

                st.write(
                    "🌐 Web Search Assistant is checking external information"
                )

                st.write(
                    "📝 Entry Agent is reconciling the final response"
                )


                try:

                    result = asyncio.run(
                        run_customer_support(
                            query
                        )
                    )


                    status.update(
                        label="Agents finished",
                        state="complete",
                        expanded=False
                    )


                except Exception as e:

                    status.update(
                        label="Processing failed",
                        state="error",
                        expanded=True
                    )

                    st.error(
                        f"An error occurred: {str(e)}"
                    )

                    st.stop()


            # =================================================
            # EXTRACT RESPONSES
            # =================================================

            responses = extract_responses(result)

            assistant_answer = responses["assistant_answer"]

            web_answer = responses["web_answer"]

            final_answer = responses["final_answer"]

            save_status = responses["save_status"]


            # =================================================
            # OUTPUT GUARDRAIL
            # =================================================

            if final_answer:

                output_check = check_output(
                    openai_client,
                    final_answer
                )

                notes = input_check.notes + output_check.notes

                if not output_check.allowed:

                    # Hide the agents' drafts too: they may carry
                    # the same unsafe content.

                    append_message(
                        "assistant",
                        output_check.reason,
                        notes=["Blocked by output guardrail."] + notes
                    )

                    render_assistant_message(
                        st.session_state.messages[-1],
                        len(st.session_state.messages) - 1
                    )

                    st.stop()

                final_answer = output_check.text

                assistant_answer, web_answer = (
                    redact_pii(text, OUTPUT_REDACT)[0] if text else text
                    for text in (assistant_answer, web_answer)
                )


            # =================================================
            # RENDER + STORE ANSWER
            # =================================================

            if final_answer:

                append_message(
                    "assistant",
                    final_answer,
                    details={
                        "assistant_answer": assistant_answer,
                        "web_answer": web_answer,
                        "saved": bool(save_status)
                    },
                    notes=notes
                )

                render_assistant_message(
                    st.session_state.messages[-1],
                    len(st.session_state.messages) - 1
                )


            else:

                st.warning(
                    "The Entry Agent did not return "
                    "a final response."
                )
