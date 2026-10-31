# Mini GPT Studio 🪻

A small, word-level GPT-style language model built from scratch with PyTorch and deployed as a Streamlit app. This project was created to learn how the core pieces of a GPT model fit together—from tokenization and embeddings to causal self-attention, a Transformer block, training, and text generation.

> This is an educational mini-model, not a production LLM. Its generations can be repetitive or grammatically inconsistent, which is expected for a small model trained on a single book.

## ✨ Try it

- **Live app:** [LINK](`https://mini-gpt-studio-madebybillubadmosh.streamlit.app/`)
- **Training notebook (Google Colab):** [Building a Mini LLM](https://colab.research.google.com/drive/1qaIPzr-TgYZ85TW68OsYExLi3N9eB0ly?authuser=2#scrollTo=lXPpEm7pkJ0r)
- **GitHub repository:** [LINK](`https://github.com/siddharthbyte/mini-gpt-studio`)

## 📚 Dataset

The model was trained on *The Child's Book of Nature* by **Worthington Hooker**, using the book's Part I section on plants as represented in the training text.

- **Original source:** [Project Gutenberg — The Child's Book of Nature](https://www.gutenberg.org/ebooks/58421)
- **Preprocessing:** text converted to lowercase; punctuation and characters outside `a-z`, `0-9`, and whitespace removed; text split on whitespace.
- **Training tokens:** 132,681 word tokens
- **Vocabulary size:** 5,308 unique words
- **Data split:** 90% training / 10% validation (119,412 training tokens and 13,269 validation tokens)

The model uses **word-level tokenization**: each unique word is assigned an integer ID. This keeps the implementation approachable, but means a prompt containing words outside the learned vocabulary cannot be represented.

## 🧠 Model architecture

| Component | Configuration |
|---|---|
| Model type | Small GPT-style causal language model |
| Tokenization | Word-level |
| Vocabulary size | 5,308 |
| Token embedding dimension | 32 |
| Positional embedding dimension | 32 |
| Context length (`block_size`) | 64 tokens |
| Transformer blocks | 1 |
| Causal self-attention heads | 1 |
| Attention head size | 32 |
| Feed-forward network | 32 → 128 → 32, with ReLU |
| Normalization | LayerNorm |
| Output layer | Linear projection to 5,308 vocabulary logits |
| Trainable parameters | **358,684** |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Training iterations | 10,000 total |
| Training hardware | Google Colab NVIDIA Tesla T4 GPU |

### How generation works

1. The prompt is lowercased and split into words.
2. Known words are converted to token IDs.
3. The model predicts next-word logits from the current context.
4. Temperature scales the logits before softmax sampling.
5. The sampled word is appended to the context and the process repeats, up to the configured generation length.

The context is limited to 64 tokens at a time. The Streamlit interface lets you enter a prompt, adjust creativity/temperature, and choose how many new tokens to generate.

## 📉 Training notes

Training was performed in two consecutive loops: 2,000 iterations followed by another 8,000 iterations. The notebook logs training and validation loss periodically.

At the final logged evaluation point (step 9,500):
- Training loss: **4.2923**
- Validation loss: **5.4997**

The notebook does not record total wall-clock training duration. Validation loss stopped improving consistently while training loss continued to decrease, so the model shows signs of overfitting to the training text.

## 🛠️ Tech stack

- Python
- PyTorch
- Streamlit
- Google Colab

## 🚀 Run locally

Clone the repository and enter the project directory:

```bash
git clone ADD_YOUR_GITHUB_REPOSITORY_URL_HERE
cd MiniLLM
```

Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the requirements and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects the trained checkpoint at `models/gpt_model.pth` and the model architecture in `model.py`. Keep those files in the locations expected by the app.

## 📁 Project structure

```text
MiniLLM/
├── app.py
├── model.py
├── test_model.py
├── requirements.txt
├── README.md
└── models/
    └── gpt_model.pth
```

## 🙌 Acknowledgements

This project was built as a learning exercise and was inspired by **Andrej Karpathy's** educational lecture:

- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

The project uses its own small word-level implementation and training corpus; it is not a reproduction of a production-scale GPT model.

## ⚠️ Limitations

- Very small model with only 358,684 trainable parameters.
- Trained on one book, so its vocabulary and style are narrow.
- Word-level tokenization cannot handle unseen words.
- Context length is limited to 64 tokens.
- Outputs are sampled predictions, not verified facts; expect occasional repetition and incoherence.
- Not suitable for factual, professional, or safety-critical use.

## 📄 Dataset rights

The linked Project Gutenberg edition identifies the book as public domain in the United States. Please check the applicable terms and copyright status for your own jurisdiction before redistributing the source text.


## 👨‍💻 Author

**Siddharth**

Built as a hands-on learning project to understand language models and Generative AI from the ground up.

---

*Small model. Big learning experience.* ✨