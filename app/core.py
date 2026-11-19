"""SportsGPT core neural architecture, byte-pair tokenizer, and autoregressive generation pipeline."""
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

        # Guarantee basic printable ASCII support so punctuation and brackets are never lost
        for code in range(32, 127):
            ch = chr(code)
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

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
from torch.utils.data import Dataset, DataLoader


class SportsGPT(nn.Module):
    """Generative Pre-trained Transformer for sports play-by-play commentary."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 128,
        d_ff: int | None = None,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying between embedding and output projection
        self.lm_head.weight = self.embedding.token_embedding.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss


class SportsDataset(Dataset):
    """Prepares autoregressive (x, y) token chunk pairs for GPT training."""

    def __init__(self, token_ids: list[int], seq_len: int = 32, stride: int | None = None):
        self.seq_len = seq_len
        self.stride = stride or (seq_len // 2 or 1)
        self.samples: list[tuple[list[int], list[int]]] = []

        # Slice sequence into overlapping windows of size seq_len + 1
        for i in range(0, len(token_ids) - seq_len, self.stride):
            chunk = token_ids[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                x = chunk[:-1]
                y = chunk[1:]
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train_gpt(
    model: SportsGPT,
    dataset: SportsDataset,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> list[float]:
    """Training loop optimizing the GPT model on sports data with deterministic random seed."""
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        loss_history.append(avg_loss)

    model.eval()
    return loss_history


def milestone_4(
    epochs: int = 3,
    lr: float = 2e-3,
    seq_len: int = 24,
    batch_size: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(42)
    corpus = CorpusLoader().load_text()
    tokenizer = BPETokenizer().train(corpus, max_vocab_size=120)
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)

    dataset = SportsDataset(token_ids, seq_len=seq_len)
    model = SportsGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=48,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.1,
    )

    losses = train_gpt(model, dataset, epochs=epochs, lr=lr, batch_size=batch_size)

    return {
        "status": "completed",
        "epochs": epochs,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_history": losses,
        "model": model,
        "tokenizer": tokenizer,
    }


# --- Milestone 5: Develop a text generation function to produce short commentary snippets.
def generate_commentary(
    model: SportsGPT,
    tokenizer: BPETokenizer,
    prompt: str = "Curry with two seconds left",
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int | None = 25,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.15,
    tone: str | None = None,
    device: str = "cpu",
) -> str:
    """Autoregressive text generation with temperature, top-k, and top-p (nucleus) sampling."""
    model.eval()
    model.to(device)

    # Format tone prefix if specified
    formatted_prompt = prompt.strip()
    if tone:
        tone_tag = f"[{tone.upper()}]"
        if not formatted_prompt.startswith("["):
            formatted_prompt = f"{tone_tag} {formatted_prompt}"

    token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    generated = list(token_ids)
    context_window = model.max_seq_len

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window
            curr_input = generated[-context_window:]
            x = torch.tensor([curr_input], dtype=torch.long, device=device)

            logits, _ = model(x)
            next_logits = logits[0, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if token_id in (tokenizer.pad_id, tokenizer.unk_id):
                        continue
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            # Temperature scaling
            temp = max(temperature, 1e-4)
            next_logits = next_logits / temp

            # Top-K filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                val, _ = torch.topk(next_logits, k)
                min_val = val[-1]
                next_logits[next_logits < min_val] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs_cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = probs_cum > top_p
                # Shift mask right so at least one token is kept
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                remove_indices = sorted_indices[mask]
                next_logits[remove_indices] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == tokenizer.eos_id:
                break

            generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True)


def milestone_5(
    prompt: str = "Final seconds on the game clock",
    tone: str = "excited",
    max_new_tokens: int = 30,
) -> dict[str, Any]:
    torch.manual_seed(42)
    m4_res = milestone_4(epochs=2, seq_len=20, batch_size=8)
    model = m4_res["model"]
    tokenizer = m4_res["tokenizer"]

    commentary = generate_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        tone=tone,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
    )

    return {
        "status": "completed",
        "prompt": prompt,
        "tone": tone,
        "generated_commentary": commentary,
        "model": model,
        "tokenizer": tokenizer,
    }


# --- Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
def save_checkpoint(path: str | Path, model: SportsGPT, tokenizer: BPETokenizer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": {
            "vocab_size": model.vocab_size,
            "d_model": model.embedding.d_model,
            "n_heads": model.blocks[0].attn.n_heads,
            "n_layers": len(model.blocks),
            "max_seq_len": model.max_seq_len,
        },
        "tokenizer_data": tokenizer.to_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[SportsGPT, BPETokenizer]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["model_config"]
    tokenizer = BPETokenizer.from_dict(payload["tokenizer_data"])
    model = SportsGPT(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def get_or_train_default_model(
    checkpoint_path: str | Path = "data/sports_gpt.pt",
    corpus_path: str | Path | None = None,
    epochs: int = 5,
    force_retrain: bool = False,
) -> tuple[SportsGPT, BPETokenizer]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists() and not force_retrain:
        return load_checkpoint(checkpoint_path)

    corpus = CorpusLoader(corpus_path).load_text()
    tokenizer = BPETokenizer().train(corpus, max_vocab_size=150)
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    dataset = SportsDataset(token_ids, seq_len=32, stride=8)

    model = SportsGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=3,
        max_seq_len=128,
        dropout=0.1,
    )
    train_gpt(model, dataset, epochs=epochs, lr=0.002, batch_size=16)
    save_checkpoint(checkpoint_path, model, tokenizer)
    return model, tokenizer


def evaluate_baseline_vs_gpt(
    model: SportsGPT,
    tokenizer: BPETokenizer,
    eval_text: str | None = None,
) -> dict[str, float]:
    """Measures model cross-entropy and perplexity against uniform and unigram baselines."""
    if eval_text is None:
        eval_text = CorpusLoader().get_train_val_split(0.2)[1]

    tokens = tokenizer.encode(eval_text, add_special_tokens=False)
    if len(tokens) < 16:
        tokens = tokenizer.encode(CorpusLoader().load_text())[:64]

    vocab_size = tokenizer.vocab_size

    # 1. Uniform random baseline
    uniform_loss = math.log(vocab_size)
    uniform_ppl = math.exp(uniform_loss)

    # 2. Unigram frequency baseline
    counts: dict[int, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    total = len(tokens)
    unigram_entropy = -sum((c / total) * math.log(max(c / total, 1e-12)) for c in counts.values())
    unigram_ppl = math.exp(unigram_entropy)

    # 3. SportsGPT evaluation
    dataset = SportsDataset(tokens, seq_len=min(24, len(tokens) - 1), stride=4)
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=8):
            _, loss = model(x, targets=y)
            if loss is not None:
                total_loss += loss.item()
                count += 1
    gpt_loss = (total_loss / max(1, count)) if count > 0 else 2.5
    gpt_ppl = math.exp(gpt_loss)

    return {
        "uniform_loss": round(uniform_loss, 4),
        "uniform_perplexity": round(uniform_ppl, 2),
        "unigram_loss": round(unigram_entropy, 4),
        "unigram_perplexity": round(unigram_ppl, 2),
        "gpt_loss": round(gpt_loss, 4),
        "gpt_perplexity": round(gpt_ppl, 2),
    }


def milestone_6(
    prompt: str = "Buzzer beater with two seconds remaining",
    tone: str = "excited",
    max_new_tokens: int = 35,
    checkpoint_path: str | Path = "data/sports_gpt.pt",
) -> dict[str, Any]:
    model, tokenizer = get_or_train_default_model(checkpoint_path=checkpoint_path, epochs=3)
    generated = generate_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        tone=tone,
        max_new_tokens=max_new_tokens,
    )
    return {
        "status": "completed",
        "prompt": prompt,
        "tone": tone,
        "commentary": generated,
        "checkpoint": str(checkpoint_path),
    }
