
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1)
        weights = weights / (self.head_size ** 0.5)

        weights = weights.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        out = weights @ v

        return out




class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)




class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()

        self.sa = SelfAttentionHead(
            embedding_dim, head_size, block_size
        )

        self.ffwd = FeedForward(embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x




class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size, embedding_dim
        )

        self.position_embedding_table = nn.Embedding(
            block_size, embedding_dim
        )

        self.block = TransformerBlock(
            embedding_dim, embedding_dim, block_size
        )

        self.ln_f = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits
