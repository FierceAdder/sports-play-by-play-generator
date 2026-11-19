# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/sm63twv772/badge.svg)](https://ask.thili.ai/b/sm63twv772)

This project develops a Sports Play-by-Play Generator, a command-line assistant for sports fans. It uses a GPT model, built from scratch and trained on real sports news, to create dynamic commentary snippets for fictional game scenarios. Dive into the world of Media & Sports by generating engaging play-by-play.

## What you are building

**Goal.** You will build a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.

**Done when.** Your project is done when you have successfully applied the byte-pair tokenizer and embeddings to prepare raw sports text, implemented causal self-attention, assembled the transformer block, and trained the GPT model from scratch to generate coherent sports commentary snippets via a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Gather a small corpus of sports commentary or news articles.**  
      Hint: Focus on collecting diverse sports text data.
- [x] **2. Implement the byte-pair tokenizer and create embeddings from the sports text.**  
      Hint: Apply the byte-pair tokenizer from the drills to process your collected text.
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Implement causal self-attention as learned, then combine it into a transformer block.
- [x] **4. Train the GPT model using the sports text data.**  
      Hint: Use the training loop from the drills to optimize your GPT model on the sports corpus.
- [x] **5. Develop a text generation function to produce short commentary snippets.**  
      Hint: Create a sampling function to generate new tokens based on the trained model's predictions.
- [x] **6. Create a CLI where users can input a prompt and receive generated commentary.**  
      Hint: Build a command-line interface to take user input and display generated text.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different sizes or types of sports text corpora to improve generation quality.
- Add options for users to specify the tone or style of commentary (e.g., excited, neutral).
- Integrate a simple mechanism to generate multiple commentary options for a single prompt.
