# 🧠 Mini GPT Studio — Built from Scratch

A miniature GPT-style language model built using **PyTorch**, trained from scratch on text data, and integrated into an interactive **Streamlit web application**.

This project explores the fundamentals of Generative AI, Transformer architecture, autoregressive text generation, and model deployment.

## ✨ Features

- **Text generation:** Enter a prompt and generate text using your trained GPT model.
- **Adjustable generation length:** Control the number of new tokens generated.
- **Temperature control:** Experiment with more focused or more random outputs.
- **Interactive interface:** Clean, dark-themed Streamlit application.
- **Example prompts:** Quickly test the model with sample inputs.
- **Local inference:** Run the model on your CPU without requiring a GPU.



## 🔗 Links

- 🚀 **Live Demo:** [Try Mini GPT Studio](https://mini-gpt-studio-madebybillubadmosh.streamlit.app/)
- 📓 **Training Notebook:** [View the Google Colab Notebook](https://colab.research.google.com/drive/1qaIPzr-TgYZ85TW68OsYExLi3N9eB0ly?authuser=2#scrollTo=lXPpEm7pkJ0r)
- 💻 **Source Code:** [View Repository](https://github.com/siddharthbyte/mini-gpt-studio)

## 🏗️ Model Architecture

The model implements a simplified GPT-style architecture with:

- Token embeddings
- Positional embeddings
- Causal self-attention
- Feed-forward neural network
- Layer normalization
- Residual connections
- Linear language-model output head

The model predicts the next word based on the preceding context, making it an **autoregressive language model**.

## 🛠️ Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy
- Git and GitHub

## 📁 Project Structure

```text
MiniLLM/
├── app.py                 # Streamlit web application
├── model.py               # GPT model architecture
├── test_model.py          # Model loading and generation test
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── models/
│   └── gpt_model.pth      # Trained model checkpoint
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mini-gpt-studio.git
cd mini-gpt-studio
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## 💬 How to Use

1. Open the application in your browser.
2. Enter a starting phrase in the prompt box.
3. Adjust the generation length and temperature in the sidebar.
4. Click **Generate text** to produce a continuation.
5. Try another prompt or use **Clear** to start again.

Example prompts:

- `the sun`
- `once upon a time`
- `in a great many`
- `the animals`

## ⚠️ Limitations

This is a small experimental language model, not a production-scale LLM.

- Generated text may be repetitive, grammatically incorrect, or incoherent.
- The model has a limited vocabulary and can only use words learned during training.
- Its knowledge and writing style are constrained by its training corpus.
- Inference runs on the CPU by default, so generation speed depends on the machine.

These limitations are expected for a small model trained for educational purposes.

## 🎯 Learning Objectives

This project helped explore:

- How tokenization and vocabulary mappings work
- How causal self-attention processes context
- How Transformer blocks are constructed
- How next-token prediction enables text generation
- How to save and reload a trained PyTorch model
- How to build a user interface around a machine-learning model

## 🔮 Future Improvements

- Improve training and generation quality
- Add model evaluation and training-loss visualizations
- Experiment with larger datasets and model architectures
- Deploy the application online
- Explore subword tokenization and more advanced sampling methods

## 👨‍💻 Author

**Siddharth**

Built as a hands-on learning project to understand language models and Generative AI from the ground up.

---

*Small model. Big learning experience.* ✨