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
        --bg: #080d1b;
        --panel: #10182b;
        --panel-2: #0b1324;
        --border: #24314e;
        --muted: #aab6d0;
        --purple: #9b5cff;
        --blue: #3478f6;
        --mint: #27e0b3;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, rgba(77, 75, 190, .16), transparent 30%),
            radial-gradient(circle at 20% 35%, rgba(33, 106, 211, .09), transparent 28%),
            var(--bg);
        color: #f5f7ff;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(8, 13, 27, .82); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1426 0%, #080e1d 100%);
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
        background: linear-gradient(145deg, rgba(18, 27, 47, .96), rgba(11, 19, 35, .96));
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
        background: #080f1e;
        border: 1px solid #22314e;
        border-radius: 13px;
        padding: 1.1rem 1.2rem;
        color: #d8e2fa;
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
        border: 1px solid #354467;
        transition: all .18s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #8e4dff, #3478f6);
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        border-color: #a27aff;
        transform: translateY(-1px);
    }
    .tip {
        border: 1px solid #40366e;
        background: linear-gradient(130deg, rgba(78, 53, 143, .25), rgba(22, 37, 75, .42));
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
        background: #0a1223 !important;
        color: #f5f7ff !important;
        border-color: #34456a !important;
        border-radius: 12px !important;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .4rem; }
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
        '<div style="font-size:2rem">✦ 🧠</div>'
        '<p class="brand">Mini <span>GPT</span></p>'
        '<p class="brand-sub">Built from scratch</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### ⚙️ Generation settings")
    max_new_tokens = st.slider(
        "Max new words",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="How many words the model should generate after your prompt.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.3,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower is more focused; higher is more random.",
    )
    st.caption("More focused  ←  Temperature  →  More creative")

    st.markdown(
        """
        <div class="tip">
        <strong>💡 Quick tip</strong><br>
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
st.markdown('<div class="eyebrow">Your personal language model</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Your Own <span class="gradient-text">GPT Model</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">A miniature GPT trained from scratch on a text corpus. '
    'Give it a starting phrase and see how it continues the text.</p>',
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">✍️ Enter your prompt</div>', unsafe_allow_html=True)
st.text_area(
    "Prompt",
    key="prompt",
    height=105,
    max_chars=500,
    placeholder="Try something like: the sun, once upon a time...",
    label_visibility="collapsed",
)

left, right = st.columns([1.5, 1], gap="medium")
with left:
    generate_clicked = st.button("✨  Generate text", type="primary", use_container_width=True)
with right:
    st.button("↻  Clear", on_click=clear_app, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


if generate_clicked:
    prompt_text = st.session_state.prompt.strip()

    if not prompt_text:
        st.warning("Enter a prompt first, then click Generate text.")
    elif not CHECKPOINT_PATH.exists():
        st.error(
            "I couldn't find the model checkpoint. Make sure this file exists: "
            "`models/gpt_model.pth` inside your MiniLLM project folder."
        )
    else:
        try:
            with st.spinner("Waking up your tiny GPT..."):
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
                        "None of the prompt words were found in this model's vocabulary. "
                        "Try common words from the training text."
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
st.markdown('<div class="section-title">🟢 Generated output</div>', unsafe_allow_html=True)

if st.session_state.generated_text:
    st.code(st.session_state.generated_text, language=None)
else:
    st.markdown(
        '<div class="output-box empty-output">Your generated text will appear here. '
        'Enter a prompt above and press Generate text to begin.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.last_note:
    st.info(st.session_state.last_note)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("### 💫 Try an example")
st.caption("Select an example to load it into the prompt box.")
examples = ["the sun", "once upon a time", "in a great many", "the animals"]
example_cols = st.columns(len(examples))
for col, example in zip(example_cols, examples):
    with col:
        st.button(
            f"↗  {example}",
            key=f"example_{example}",
            use_container_width=True,
            on_click=use_example,
            args=(example,),
        )

st.markdown(
    "<div class='footer'>Mini GPT Studio · A small model, built from scratch, for learning and exploration.</div>",
    unsafe_allow_html=True,
)
