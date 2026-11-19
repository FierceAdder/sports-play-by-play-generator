"""Comprehensive test suite verifying SportsGPT architecture, tokenizer, training, and generation."""
import pytest

# Milestone 1: Gather a small corpus of sports commentary or news articles.
def test_milestone_1():
    from app.core import CorpusLoader, milestone_1

    loader = CorpusLoader()
    text = loader.load_text()
    assert len(text) > 5000, "Corpus should contain substantial sports text"

    stats = loader.get_stats()
    assert stats["total_words"] > 500
    assert len(stats["detected_sports"]) >= 4, "Corpus should cover multiple diverse sports"
    assert "EXCITED" in stats["detected_tones"]
    assert "NEUTRAL" in stats["detected_tones"]

    train_txt, val_txt = loader.get_train_val_split(0.1)
    assert len(train_txt) > 0 and len(val_txt) > 0

    res = milestone_1()
    assert res["status"] == "completed"
    assert "total_words" in res["stats"]


# Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
def test_milestone_2():
    import torch
    from app.core import BPETokenizer, TransformerEmbedding, milestone_2

    tokenizer = BPETokenizer()
    training_sample = "Curry hits a deep three! Touchdown Kansas City! Game over!"
    tokenizer.train(training_sample, max_vocab_size=80)
    assert tokenizer.vocab_size > 10

    encoded = tokenizer.encode(training_sample, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)
    assert decoded == training_sample

    d_model = 32
    emb_layer = TransformerEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model, max_seq_len=64)
    inp = torch.tensor([encoded], dtype=torch.long)
    out = emb_layer(inp)
    assert out.shape == (1, len(encoded), d_model)
    assert not torch.isnan(out).any()

    # Verify serialization
    tok_dict = tokenizer.to_dict()
    reloaded = BPETokenizer.from_dict(tok_dict)
    assert reloaded.decode(reloaded.encode("Curry hits")) == "Curry hits"

    res = milestone_2(sample_text="Goal in stoppage time!")
    assert res["status"] == "completed"
    assert res["embeddings_shape"][2] == 64


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def test_milestone_3():
    import torch
    from app.core import CausalSelfAttention, TransformerBlock, milestone_3

    d_model = 32
    n_heads = 4
    seq_len = 12
    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=32, dropout=0.0)
    attn.eval()

    x = torch.randn(2, seq_len, d_model)
    out = attn(x)
    assert out.shape == (2, seq_len, d_model)

    # Check causality strictly: changing future token doesn't affect past output
    x_perturbed = x.clone()
    x_perturbed[:, 7:, :] += 10.0
    out_perturbed = attn(x_perturbed)
    diff = torch.max(torch.abs(out[:, :7, :] - out_perturbed[:, :7, :])).item()
    assert diff < 1e-5, f"Causal violation: past tokens changed by future input (diff={diff})"

    block = TransformerBlock(d_model=d_model, n_heads=n_heads, max_seq_len=32, dropout=0.0)
    block_out = block(x)
    assert block_out.shape == (2, seq_len, d_model)

    res = milestone_3()
    assert res["status"] == "completed"
    assert res["is_causal"] is True
    assert res["past_difference"] < 1e-5


# Milestone 4: Train the GPT model using the sports text data.
def test_milestone_4():
    from app.core import SportsGPT, SportsDataset, train_gpt, BPETokenizer, milestone_4

    sample_corpus = (
        "Curry hits the three! Touchdown Kansas City! What a goal in stoppage time! "
        "A walk-off home run into the upper deck! Match point for Alcaraz on Centre Court! "
    ) * 4
    tokenizer = BPETokenizer().train(sample_corpus, max_vocab_size=60)
    tokens = tokenizer.encode(sample_corpus)

    dataset = SportsDataset(tokens, seq_len=16, stride=4)
    assert len(dataset) > 0

    model = SportsGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=2,
        max_seq_len=32,
    )

    # Initial forward check
    x, y = dataset[0]
    logits, loss = model(x.unsqueeze(0), y.unsqueeze(0))
    assert logits.shape == (1, 16, tokenizer.vocab_size)
    assert loss is not None and loss.item() > 0

    losses = train_gpt(model, dataset, epochs=3, lr=0.005, batch_size=4)
    assert len(losses) == 3
    assert losses[-1] < losses[0], "Training loss should decrease"

    res = milestone_4(epochs=2)
    assert res["status"] == "completed"
    assert res["final_loss"] <= res["initial_loss"]


# Milestone 5: Develop a text generation function to produce short commentary snippets.
def test_milestone_5():
    from app.core import (
        BPETokenizer,
        SportsGPT,
        SportsDataset,
        train_gpt,
        generate_commentary,
        milestone_5,
    )

    corpus = "[EXCITED] Curry hits a deep buzzer beater! Touchdown Kansas City! What a goal!" * 5
    tokenizer = BPETokenizer().train(corpus, max_vocab_size=150)
    dataset = SportsDataset(tokenizer.encode(corpus), seq_len=12, stride=3)
    model = SportsGPT(vocab_size=tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=60)
    train_gpt(model, dataset, epochs=2, batch_size=4)

    prompt = "Curry"
    gen_text = generate_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=15,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
    )
    assert len(gen_text) >= len(prompt)
    assert prompt in gen_text

    # Test tone conditioning
    toned_text = generate_commentary(
        model=model,
        tokenizer=tokenizer,
        prompt="Touchdown",
        tone="excited",
        max_new_tokens=10,
    )
    assert "[EXCITED]" in toned_text

    res = milestone_5(prompt="Curry", max_new_tokens=10)
    assert res["status"] == "completed"
    assert len(res["generated_commentary"]) > 0


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
def test_milestone_6(capsys, tmp_path):
    from app.core import milestone_6
    from app.cli import main

    ckpt = tmp_path / "test_gpt.pt"
    res = milestone_6(
        prompt="Final seconds on the game clock",
        tone="excited",
        max_new_tokens=15,
        checkpoint_path=ckpt,
    )
    assert res["status"] == "completed"
    assert len(res["commentary"]) > 0
    assert ckpt.exists()

    # Test CLI generate
    main(["generate", "--prompt", "Curry at the buzzer", "--max-tokens", "10", "--checkpoint", str(ckpt)])
    out = capsys.readouterr().out
    assert "Sports Play-by-Play Commentary" in out
    assert "Curry at the buzzer" in out

    # Test CLI evaluate
    main(["evaluate", "--checkpoint", str(ckpt)])
    eval_out = capsys.readouterr().out
    assert "Model vs Baseline Comparison" in eval_out
    assert "SportsGPT" in eval_out

    # Test CLI sample-options
    main(["sample-options", "--prompt", "Match point", "-n", "2", "--checkpoint", str(ckpt)])
    opts_out = capsys.readouterr().out
    assert "Option 1" in opts_out
    assert "Option 2" in opts_out


def test_checkpoint_save_and_load(tmp_path):
    from app.core import SportsGPT, BPETokenizer, save_checkpoint, load_checkpoint

    tok = BPETokenizer()
    tok.train("Curry scores buzzer beater", max_vocab_size=25)
    model = SportsGPT(vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, max_seq_len=20)

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tok)
    assert path.exists()

    loaded_model, loaded_tok = load_checkpoint(path)
    assert loaded_model.vocab_size == model.vocab_size
    assert loaded_tok.vocab_size == tok.vocab_size


def test_baseline_evaluation():
    from app.core import SportsGPT, BPETokenizer, SportsDataset, train_gpt, evaluate_baseline_vs_gpt

    tok = BPETokenizer()
    text = "Curry hits a three! Goal in stoppage! Touchdown! Match point!" * 6
    tok.train(text, max_vocab_size=40)
    dataset = SportsDataset(tok.encode(text), seq_len=16, stride=4)
    model = SportsGPT(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=2, max_seq_len=30)
    train_gpt(model, dataset, epochs=3, lr=0.01, batch_size=4)

    metrics = evaluate_baseline_vs_gpt(model, tok, eval_text=text)

    assert "uniform_loss" in metrics
    assert "unigram_loss" in metrics
    assert "gpt_loss" in metrics
    assert metrics["uniform_perplexity"] > 0
    assert metrics["gpt_perplexity"] > 0
    assert metrics["gpt_loss"] < metrics["uniform_loss"]

