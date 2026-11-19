"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Train the GPT model using the sports text data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Develop a text generation function to produce short commentary snippets.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

