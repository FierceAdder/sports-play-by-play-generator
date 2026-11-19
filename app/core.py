"""Your code. One stub per milestone; build them in order and keep each one small."""
import re
from pathlib import Path
from typing import Any


class CorpusLoader:
    """Loads and analyzes a small sports commentary or news corpus."""

    DEFAULT_PATH = Path("data/sports_corpus.txt")

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else self.DEFAULT_PATH

    def load_text(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Sports corpus not found at {self.path}")
        return self.path.read_text(encoding="utf-8").strip()

    def get_paragraphs(self) -> list[str]:
        text = self.load_text()
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def get_stats(self) -> dict[str, Any]:
        text = self.load_text()
        paragraphs = self.get_paragraphs()
        words = re.findall(r"\b\w+\b", text)
        tones = sorted(set(re.findall(r"\[(EXCITED|NEUTRAL|DRAMATIC)\]", text)))

        # Detect sport keywords
        sports_keywords = {
            "basketball": ["curry", "tatum", "rebound", "three-pointer", "buzzer-beater", "dunk"],
            "soccer": ["goal", "box", "midfield", "crossbar", "saka", "bellingham", "stoppage"],
            "football": ["touchdown", "quarterback", "mahomes", "first and ten", "yard line"],
            "tennis": ["baseline", "break point", "forehand", "djokovic", "alcaraz"],
            "baseball": ["inning", "grand slam", "pitcher", "ohtani", "batter"],
            "hockey": ["puck", "ice", "winger", "five-hole"],
        }
        text_lower = text.lower()
        detected_sports = [
            sport for sport, kws in sports_keywords.items() if any(kw in text_lower for kw in kws)
        ]

        return {
            "total_characters": len(text),
            "total_words": len(words),
            "total_paragraphs": len(paragraphs),
            "unique_words": len(set(w.lower() for w in words)),
            "detected_sports": detected_sports,
            "detected_tones": tones,
        }

    def get_train_val_split(self, val_ratio: float = 0.1) -> tuple[str, str]:
        paragraphs = self.get_paragraphs()
        split_idx = max(1, int(len(paragraphs) * (1 - val_ratio)))
        train_text = "\n\n".join(paragraphs[:split_idx])
        val_text = "\n\n".join(paragraphs[split_idx:])
        return train_text, val_text


# --- Milestone 1: Gather a small corpus of sports commentary or news articles.
# Hint: Focus on collecting diverse sports text data.
def milestone_1(corpus_path: str | Path | None = None) -> dict[str, Any]:
    loader = CorpusLoader(corpus_path)
    stats = loader.get_stats()
    text = loader.load_text()
    return {
        "status": "completed",
        "stats": stats,
        "sample": text[:200] + "...",
    }


# --- Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
# Hint: Apply the byte-pair tokenizer from the drills to process your collected text.
import json
import torch
import torch.nn as nn


class BPETokenizer:
    """Byte-Pair Encoding (BPE) tokenizer learned from corpus text."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self):
        self.special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_token: dict[int, str] = {idx: tok for idx, tok in enumerate(self.special_tokens)}
        self.merges: list[tuple[str, str]] = []

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train(self, text: str, max_vocab_size: int = 150, min_freq: int = 2) -> "BPETokenizer":
        # Initial character alphabet
        chars = sorted(set(text))
        for ch in chars:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        # Represent words as tuples of symbols
        words = text.split()
        word_freqs: dict[tuple[str, ...], int] = {}
        for w in words:
            if not w:
                continue
            seq = tuple(list(w))
            word_freqs[seq] = word_freqs.get(seq, 0) + 1

        while len(self.token_to_id) < max_vocab_size:
            pair_counts: dict[tuple[str, str], int] = {}
            for word_tuple, count in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                break

            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], len(kv[0][0]) + len(kv[0][1])))
            pair, count = best_pair
            if count < min_freq:
                break

            merged_token = pair[0] + pair[1]
            if merged_token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[merged_token] = new_id
                self.id_to_token[new_id] = merged_token
                self.merges.append(pair)

            # Update words with merged pair
            new_word_freqs = {}
            for word_tuple, c in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = c
            word_freqs = new_word_freqs

        return self

    def _tokenize_word(self, word: str) -> list[str]:
        tokens = list(word)
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens: list[str] = []
        words = text.split(" ")
        for i, word in enumerate(words):
            if i > 0:
                tokens.append(" ")
            tokens.extend(self._tokenize_word(word))

        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            tok = self.id_to_token.get(idx, self.UNK)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPETokenizer":
        tok = cls()
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok


class TransformerEmbedding(nn.Module):
    """Combines token and learned positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int = 64, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        return self.dropout(self.layer_norm(token_emb + pos_emb))


def milestone_2(
    corpus_text: str | None = None,
    sample_text: str = "[EXCITED] Curry buzzer-beater at the buzzer!",
    max_vocab_size: int = 150,
    d_model: int = 64,
) -> dict[str, Any]:
    if corpus_text is None:
        corpus_text = CorpusLoader().load_text()
    tokenizer = BPETokenizer().train(corpus_text, max_vocab_size=max_vocab_size)
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)

    embedding_layer = TransformerEmbedding(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        max_seq_len=max(128, len(encoded) + 10),
    )
    tokens_tensor = torch.tensor([encoded], dtype=torch.long)
    embeddings = embedding_layer(tokens_tensor)

    return {
        "status": "completed",
        "vocab_size": tokenizer.vocab_size,
        "encoded_tokens": encoded,
        "decoded_text": decoded,
        "embeddings_shape": list(embeddings.shape),
        "tokenizer": tokenizer,
        "embedding_layer": embedding_layer,
    }


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Implement causal self-attention as learned, then combine it into a transformer block.
import math
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Joint projection for Query, Key, and Value
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular causal mask buffer
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: mask upper triangle to -infinity
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context aggregation
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int = 64, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block combining causal self-attention and MLP."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        max_seq_len: int = 256,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def milestone_3(
    batch_size: int = 2,
    seq_len: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
    max_seq_len: int = 64,
) -> dict[str, Any]:
    torch.manual_seed(42)
    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    block.eval()

    # Create dummy input
    x1 = torch.randn(batch_size, seq_len, d_model)
    out1 = block(x1)

    # Causality verification: perturbing future position t=10 must not alter output at t < 10
    x2 = x1.clone()
    x2[:, 10:, :] = torch.randn_like(x2[:, 10:, :])
    out2 = block(x2)

    past_diff = torch.max(torch.abs(out1[:, :10, :] - out2[:, :10, :])).item()
    is_causal = past_diff < 1e-5

    return {
        "status": "completed",
        "output_shape": list(out1.shape),
        "is_causal": is_causal,
        "past_difference": past_diff,
        "transformer_block": block,
    }


# --- Milestone 4: Train the GPT model using the sports text data.
# Hint: Use the training loop from the drills to optimize your GPT model on the sports corpus.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Develop a text generation function to produce short commentary snippets.
# Hint: Create a sampling function to generate new tokens based on the trained model's predictions.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
# Hint: Build a command-line interface to take user input and display generated text.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
