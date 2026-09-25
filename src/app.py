import streamlit as st
import ollama
import sqlite3
import time
import uuid
import router as rtr
import consolidate
import dedup

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = "data/memory.db"
MODEL = "qwen2.5:1.5b"  # matched to the rest of the project for consistency

st.set_page_config(
    page_title="Localized Agentic Memory",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# CUSTOM STYLING (unchanged from Usman's version)
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #1A1A1A;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #1A1A1A !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

.title-text {
    font-size: 2.2rem;
    font-weight: 700;
    color: #F44336;
    margin-bottom: 0.1rem;
}

.tagline {
    color: #5B6470;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

.status-bar {
    display: inline-flex;
    align-items: center;
    background: #DCE8DF;
    padding: 8px 16px;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    border: 1px solid #3B7A57;
}

.status-bar strong {
    color: #1F2937;
}

.status-dot {
    height: 10px;
    width: 10px;
    background-color: #3B7A57;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.mode-strip {
    background: #EAE7E1;
    border-left: 4px solid #3B7A57;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 15px;
}

.user-message {
    background-color: #DCE8DF;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
    color: #1F2937;
}

.assistant-message {
    background-color: #FFFFFF;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    border: 1px solid #E5E7EB;
    color: #1F2937;
}

.footer {
    text-align: center;
    color: #6B7280;
    margin-top: 25px;
    font-size: 0.85rem;
}

/* ---------- Text input & mode buttons (light theme) ---------- */
div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #D8D3C7 !important;
    color: #1F2937 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
}

/* Mode toggle buttons */
div[data-testid="column"] .stButton button {
    border-radius: 20px;
    border: 1px solid #D8D3C7;
    background: #FFFFFF;
    color: #5B6470;
    font-size: 12.5px;
    padding: 4px 14px;
}

div[data-testid="column"] .stButton button:hover {
    border-color: #3B7A57;
    color: #3B7A57;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# BACKEND WIRING — persisted across Streamlit reruns via session_state
# -----------------------------
if "conn" not in st.session_state:
    st.session_state.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "chat"

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "chat"


def save_turn(conn, session_id, role, content):
    cursor = conn.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    return cursor.lastrowid


def delete_turns(conn, ids):
    """Removes the given conversation rows entirely — used when the user
    marks a response as not worth remembering. Since nothing gets consolidated
    until session exit, deleting here guarantees it never reaches long-term
    memory at all."""
    ids = [i for i in ids if i is not None]
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"DELETE FROM conversations WHERE id IN ({placeholders})", tuple(ids))
    conn.commit()


def handle_chat(cleaned_input):
    history = rtr.load_history(
        st.session_state.conn, st.session_state.session_id)
    response = ollama.chat(model=MODEL, messages=history)
    return response["message"]["content"]


handlers = {
    "search": rtr.handle_search,
    "deepsearch": rtr.handle_deepsearch,
    "remember": rtr.handle_remember,
    "incognito": rtr.handle_incognito,
    "chat": handle_chat,
}

# -----------------------------
# TITLE AREA
# -----------------------------
st.markdown(
    '<div class="title-text">Localized Agentic Memory</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">'
    'Runs entirely on this device — no cloud, no account, nothing leaves this laptop.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="status-bar">
    <span class="status-dot"></span>
    <strong>System Ready</strong>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# CHAT HISTORY (with per-turn timing, so "fast" is shown, not just claimed)
# -----------------------------
chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if "timing" in msg:
                st.markdown(
                    f'<div class="timing-caption">buffer write: {msg["timing"]["write_ms"]:.2f}ms '
                    f'&nbsp;|&nbsp; model inference: {msg["timing"]["infer_ms"]:.1f}ms</div>',
                    unsafe_allow_html=True
                )

            # Feedback — only shown for turns that were actually saved
            # (skipped for incognito, and hidden once a decision's been made).
            has_saved_ids = msg.get("user_msg_id") is not None or msg.get(
                "assistant_msg_id") is not None
            if has_saved_ids and not msg.get("feedback_given"):
                fcol1, fcol2, fspacer = st.columns([1, 1, 8])
                with fcol1:
                    if st.button("👍", key=f"good_{i}"):
                        st.session_state.messages[i]["feedback_given"] = "good"
                        st.rerun()
                with fcol2:
                    if st.button("👎", key=f"bad_{i}"):
                        delete_turns(
                            st.session_state.conn,
                            [msg.get("user_msg_id"), msg.get(
                                "assistant_msg_id")]
                        )
                        st.session_state.messages[i]["feedback_given"] = "bad"
                        st.rerun()
            elif msg.get("feedback_given") == "bad":
                st.markdown(
                    '<div class="timing-caption">Marked unhelpful — not saved to memory</div>',
                    unsafe_allow_html=True
                )

# -----------------------------
# MODE SELECTOR + INPUT — one unified row, no st.form().
# A <form> only supports ONE unambiguous submit action; mixing mode-select
# buttons with a submit button inside a form made Enter presses trigger the
# wrong button unpredictably. Using a plain text_input with an on_change
# callback (fires on Enter) removes that ambiguity entirely, and frees the
# layout to put everything in one row like the reference design.
# -----------------------------
MODE_OPTIONS = [
    ("chat", "💬 Chat"),
    ("search", "🔍 Search"),
    ("deepsearch", "📖 Deepsearch"),
    ("remember", "🧠 Remember"),
    ("incognito", "🕶️ Incognito"),
]


def submit_message(user_input):
    user_input = user_input.strip()
    if not user_input:
        return

    # If the user manually typed a prefix, respect it (keeps CLI-style testing
    # working). Otherwise, fall back to whichever mode button is selected.
    detected_mode, detected_clean = rtr.route_message(user_input)
    if detected_mode != "chat":
        mode, cleaned_input = detected_mode, detected_clean
    else:
        mode, cleaned_input = st.session_state.selected_mode, user_input

    st.session_state.mode = mode
    st.session_state.messages.append({"role": "user", "content": user_input})

    write_ms = 0.0
    user_msg_id = None
    if mode != "incognito":
        start = time.perf_counter()
        user_msg_id = save_turn(st.session_state.conn,
                                st.session_state.session_id, "user", user_input)
        write_ms += (time.perf_counter() - start) * 1000

    handler_func = handlers.get(mode, handle_chat)
    infer_start = time.perf_counter()
    with st.spinner("Thinking…"):
        try:
            assistant_reply = handler_func(cleaned_input)
        except Exception as e:
            assistant_reply = f"Sorry, I encountered an error while processing your request. ({e})"
    infer_ms = (time.perf_counter() - infer_start) * 1000

    assistant_msg_id = None
    if mode != "incognito" and assistant_reply.strip() != "":
        start = time.perf_counter()
        assistant_msg_id = save_turn(
            st.session_state.conn, st.session_state.session_id, "assistant", assistant_reply)
        write_ms += (time.perf_counter() - start) * 1000

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply,
        "timing": {"write_ms": write_ms, "infer_ms": infer_ms},
        "user_msg_id": user_msg_id,
        "assistant_msg_id": assistant_msg_id,
    })


def on_enter():
    # Fires when Enter is pressed in the text input (or it loses focus).
    submit_message(st.session_state.chat_input_box)
    st.session_state.chat_input_box = ""  # safe here — inside a callback


st.text_input(
    "Message",
    placeholder="Type your message...",
    label_visibility="collapsed",
    key="chat_input_box",
    on_change=on_enter,
)

row_cols = st.columns(len(MODE_OPTIONS) + 1)
for i, (mode_key, mode_label) in enumerate(MODE_OPTIONS):
    with row_cols[i]:
        is_active = st.session_state.selected_mode == mode_key
        if st.button(
            mode_label,
            key=f"mode_btn_{mode_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_mode = mode_key
            st.rerun()

with row_cols[-1]:
    if st.button("➜ Send", type="primary", use_container_width=True):
        submit_message(st.session_state.chat_input_box)
        st.session_state.chat_input_box = ""
        st.rerun()

# -----------------------------
# END SESSION — triggers real consolidation + dedup
# -----------------------------
st.markdown("---")
if st.button("End Session & Save Memory"):
    consolidate.trigger_consolidation(
        st.session_state.conn, st.session_state.session_id)
    dedup.trigger_deduplication(st.session_state.conn)
    st.success(
        "Session consolidated — facts extracted, deduplicated, and saved to long-term memory.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    '<div class="footer">'
    'Localized Agentic Memory Framework — Final Year Project'
    '</div>',
    unsafe_allow_html=True
)
