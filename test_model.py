
import torch
from model import GPTLanguageModel

checkpoint = torch.load(
    "models/gpt_model.pth",
    map_location="cpu",
    weights_only=False
)

model = GPTLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    block_size=checkpoint["block_size"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
print("Vocabulary size:", checkpoint["vocab_size"])




import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens=30, temperature=0.7):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        probs = F.softmax(logits / temperature, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


prompt = "the sun"
prompt_ids = [checkpoint["stoi"][word] for word in prompt.split()]

context = torch.tensor([prompt_ids], dtype=torch.long)

generated_ids = generate(model, context, max_new_tokens=30)

generated_words = [
    checkpoint["itos"][token_id]
    for token_id in generated_ids[0].tolist()
]

print("\nGenerated text:")
print(" ".join(generated_words))
