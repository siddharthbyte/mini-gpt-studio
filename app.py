import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F

from model import GPTLanguageModel


# -----------------------------
# Page configuration and styling
# -----------------------------
st.set_page_config(
    page_title="Mini GPT Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #101321;
        --panel: #191b30;
        --panel-2: #14172a;
        --border: #343653;
        --muted: #b8bad4;
        --purple: #b5a4ff;
        --blue: #8db8ff;
        --mint: #27e0b3;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, rgba(157, 139, 255, .17), transparent 30%),
            radial-gradient(circle at 20% 35%, rgba(100, 161, 255, .10), transparent 28%),
            var(--bg);
        color: #f5f7ff;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(16, 19, 33, .88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1c32 0%, #111423 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.2rem; }

    .brand {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .brand span {
        background: linear-gradient(90deg, #bd63ff, #4f91ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub { color: var(--muted); margin-top: 0; font-size: .92rem; }
    .eyebrow {
        color: #9d8bff;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .hero-title {
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -1.8px;
        margin: 0;
    }
    .gradient-text {
        background: linear-gradient(90deg, #bb6cff, #5793ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #b2bed8;
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 760px;
        margin-top: .8rem;
        margin-bottom: 1.5rem;
    }
    .panel {
        background: linear-gradient(145deg, rgba(30, 32, 55, .96), rgba(22, 25, 43, .97));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, .12);
    }
    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: .8rem;
    }
    .small-muted { color: var(--muted); font-size: .88rem; }
    .output-box {
        background: #141729;
        border: 1px solid #353955;
        border-radius: 13px;
        padding: 1.1rem 1.2rem;
        color: #e5e5fa;
        font-family: 'DM Mono', monospace;
        line-height: 1.9;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        min-height: 150px;
    }
    .empty-output {
        color: #8391af;
        font-family: 'Inter', sans-serif;
        font-style: italic;
    }
    div.stButton > button {
        border-radius: 12px;
        min-height: 2.85rem;
        font-weight: 700;
        border: 1px solid #414563;
        transition: all .18s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #a58aff, #75a9ff);
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        border-color: #b7a5ff;
        transform: translateY(-1px);
    }
    .tip {
        border: 1px solid #514b7b;
        background: linear-gradient(130deg, rgba(117, 93, 184, .20), rgba(63, 93, 151, .22));
        border-radius: 14px;
        padding: .9rem;
        color: #c8d1eb;
        font-size: .86rem;
        line-height: 1.65;
    }
    .footer {
        color: #8794b0;
        text-align: center;
        font-size: .82rem;
        padding: 1.4rem 0 .4rem;
    }
    .stTextArea textarea, .stTextInput input {
        background: #171a2d !important;
        color: #f5f7ff !important;
        border-color: #464968 !important;
        border-radius: 12px !important;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .4rem; }
    [data-testid="stSlider"] [role="slider"] { background: #b7a5ff; border-color: #b7a5ff; }
    .stTextArea textarea:focus { border-color: #a99aff !important; box-shadow: 0 0 0 1px #a99aff !important; }
    @media (max-width: 700px) {
        .hero-title { letter-spacing: -1px; }
        .panel { padding: 1rem; border-radius: 15px; }
        .brand { font-size: 1.55rem; }
    }

    /* Pastel, cozy mode */
    :root {
        --bg: #f7f4ff;
        --panel: #ffffff;
        --panel-2: #f0edff;
        --border: #e2dcfa;
        --muted: #706b8b;
        --purple: #8c73dc;
        --blue: #639be8;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(194,178,255,.48), transparent 26%),
            radial-gradient(circle at 92% 18%, rgba(164,211,255,.46), transparent 27%),
            radial-gradient(circle at 75% 88%, rgba(255,220,235,.35), transparent 23%),
            #f7f4ff !important;
        color: #302b4a !important;
        font-family: 'Nunito', 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(247,244,255,.85) !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#eee9ff 0%,#eaf3ff 100%) !important;
        border-right: 1px solid #dcd5f5;
    }
    .brand { color: #302b4a; }
    .brand-sub, .hero-sub { color: #706b8b !important; }
    .eyebrow { color: #8068c8; }
    .hero-title, .section-title { color: #393251; }
    .gradient-text {
        background: linear-gradient(90deg,#9671df,#5d9ee9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .panel {
        background: rgba(255,255,255,.86) !important;
        border: 1px solid #e2dcfa !important;
        border-radius: 22px !important;
        box-shadow: 0 12px 32px rgba(106,91,163,.09) !important;
        backdrop-filter: blur(8px);
    }
    .output-box {
        background: #f8f7ff !important;
        border: 1px dashed #cfc5f2 !important;
        color: #393251 !important;
        border-radius: 16px !important;
    }
    .empty-output { color: #8b86a5 !important; }
    .tip {
        border: 1px solid #d8cef8 !important;
        background: linear-gradient(130deg,#f7f1ff,#edf6ff) !important;
        color: #5d5579 !important;
        border-radius: 16px !important;
    }
    .footer { color: #8b85a4 !important; }
    div.stButton > button {
        border-radius: 14px !important;
        color: #51486f !important;
        background: rgba(255,255,255,.86) !important;
        border: 1px solid #dcd5f5 !important;
        font-weight: 800 !important;
        box-shadow: 0 3px 9px rgba(91,77,144,.05);
        transition: all .18s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg,#a58aee,#7eb7f2) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 7px 16px rgba(130,112,215,.2);
    }
    div.stButton > button:hover {
        border-color: #a99be5 !important;
        transform: translateY(-2px) rotate(-.3deg);
    }
    .stTextArea textarea, .stTextInput input {
        background: #fbfaff !important;
        color: #393251 !important;
        border-color: #dcd5f5 !important;
        border-radius: 14px !important;
    }
    .stTextArea textarea::placeholder { color: #a39db9 !important; }
    [data-testid="stSlider"] [role="slider"] {
        background: #947de0 !important;
        border-color: #947de0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #51486f; }

    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model loading and generation
# -----------------------------
CHECKPOINT_PATH = Path(__file__).parent / "models" / "gpt_model.pth"


@st.cache_resource
def load_model(checkpoint_path: str):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = GPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        block_size=checkpoint["block_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def generate(model, starting_ids, max_new_tokens: int, temperature: float):
    idx = torch.tensor([starting_ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 0.05)
        probabilities = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    return idx[0].tolist()


def clear_app():
    st.session_state.prompt = ""
    st.session_state.generated_text = ""
    st.session_state.last_note = ""
    st.session_state.output_display = ""


def use_example(example: str):
    st.session_state.prompt = example
    st.session_state.generated_text = ""
    st.session_state.last_note = ""


if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""
if "last_note" not in st.session_state:
    st.session_state.last_note = ""


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        '<div style="font-size:2rem">🪻 ✨</div>'
        '<p class="brand">Mini <span>GPT</span></p>'
        '<p class="brand-sub">Small brain. Big dreams.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### ✨ Make it yours")
    max_new_tokens = st.slider(
        "Max new words",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="How many words the model should generate after your prompt.",
    )
    temperature = st.slider(
        "Creativity",
        min_value=0.3,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower stays focused; higher takes more unexpected turns.",
    )
    st.caption("🌿 Chill  ·  ☁️ Wander  ·  🔥 Go wild")

    st.markdown(
        """
        <div class="tip">
        <strong>Little tip</strong><br>
        This is a small word-level GPT trained from scratch. Try short prompts
        using familiar words. The output can be quirky—that's part of the
        experiment!
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div class='footer'>Made with ❤️ for learning<br>"
        "A miniature language model experiment</div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# Main workspace
# -----------------------------
st.markdown('<div class="eyebrow">Warning: tiny brain at work</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Your very own <span class="gradient-text">word goblin</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">I trained a tiny language model on a book. '
    'Give it a few words and let the little guy cook. 👨‍🍳</p>',
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Feed me a prompt 🍪</div>', unsafe_allow_html=True)
st.text_area(
    "Prompt",
    key="prompt",
    height=105,
    max_chars=500,
    placeholder="the sun... a brave potato... once upon a time...",
    label_visibility="collapsed",
)

left, right = st.columns([1.5, 1], gap="medium")
with left:
    generate_clicked = st.button("Let it cook ✨", type="primary", use_container_width=True)
with right:
    st.button("Brain reset ↺", on_click=clear_app, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


if generate_clicked:
    prompt_text = st.session_state.prompt.strip()

    if not prompt_text:
        st.warning("The word goblin needs a snack first. Type a prompt above! 🍪")
    elif not CHECKPOINT_PATH.exists():
        st.error(
            "I couldn't find the model checkpoint. Make sure this file exists: "
            "`models/gpt_model.pth` inside your MiniLLM project folder."
        )
    else:
        try:
            with st.spinner("Giving the tiny brain a little coffee... ☕"):
                model, checkpoint = load_model(str(CHECKPOINT_PATH))

                # The model uses a word-level vocabulary and was trained on lowercase text.
                prompt_words = re.findall(r"[a-z0-9]+", prompt_text.lower())
                known_words = [
                    word for word in prompt_words
                    if word in checkpoint["stoi"]
                ]
                skipped_words = [
                    word for word in prompt_words
                    if word not in checkpoint["stoi"]
                ]

                if not known_words:
                    st.session_state.generated_text = ""
                    st.session_state.output_display = ""
                    st.session_state.last_note = (
                        "Huhhhhh? 🐱\nTry adding something from my vocab!"
                    )
                else:
                    prompt_ids = [checkpoint["stoi"][word] for word in known_words]
                    generated_ids = generate(
                        model,
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    generated_words = [
                        checkpoint["itos"][int(token_id)]
                        for token_id in generated_ids
                    ]
                    st.session_state.generated_text = " ".join(generated_words)
                    st.session_state.output_display = st.session_state.generated_text

                    if skipped_words:
                        st.session_state.last_note = (
                            "Some words aren't in the model's vocabulary and were skipped: "
                            + ", ".join(skipped_words)
                        )
                    else:
                        st.session_state.last_note = ""
        except Exception as error:
            st.error(f"Something went wrong while generating text: {error}")


st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Fresh from the word oven 🍞</div>', unsafe_allow_html=True)

if st.session_state.generated_text:
    st.code(st.session_state.generated_text, language=None)
else:
    st.markdown(
        '<div class="output-box empty-output">Your words will show up here. '
        'Start with a short phrase and see where it goes.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.last_note:
    if st.session_state.last_note.startswith("Huhhhhh?"):
        st.markdown(
            '<div class="tip" style="margin-top:.7rem">'
            '<div style="font-size:1.1rem;font-weight:700">Huhhhhh? 🐱</div>'
            '<div style="margin-top:.25rem">Try adding something from my vocab!</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("ℹ️ " + st.session_state.last_note)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("### No ideas? Steal one of these 😌")
st.caption("These are approved snacks for the model.")
examples = ["the sun", "once upon a time", "in a great many", "the animals"]
example_cols = st.columns(len(examples))
for col, example in zip(example_cols, examples):
    with col:
        st.button(
            f"{example}  ↗",
            key=f"example_{example}",
            use_container_width=True,
            on_click=use_example,
            args=(example,),
        )

st.markdown(
    "<div class='footer'>Mini GPT Studio · tiny model, questionable poetry, excellent vibes. 🪻</div>",
    unsafe_allow_html=True,
)
