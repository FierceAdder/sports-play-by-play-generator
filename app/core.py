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
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Implement causal self-attention as learned, then combine it into a transformer block.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


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
