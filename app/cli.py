"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from pathlib import Path

from app.core import (
    get_or_train_default_model,
    generate_commentary,
    evaluate_baseline_vs_gpt,
    train_gpt,
    save_checkpoint,
    CorpusLoader,
    BPETokenizer,
    SportsDataset,
    SportsGPT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sports-play-by-play-generator",
        description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Hello command (starter)
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. Generate command
    gen_parser = sub.add_parser("generate", help="generate dynamic sports play-by-play commentary")
    gen_parser.add_argument("--prompt", "-p", type=str, default="Final seconds on the game clock", help="fictional game scenario or play prompt")
    gen_parser.add_argument("--tone", "-t", type=str, choices=["excited", "neutral", "dramatic"], default="excited", help="tone or style of commentary")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (lower = more deterministic)")
    gen_parser.add_argument("--top-k", type=int, default=25, help="top-k sampling threshold")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="top-p nucleus sampling threshold")
    gen_parser.add_argument("--max-tokens", "-m", type=int, default=45, help="maximum new tokens to generate")
    gen_parser.add_argument("--checkpoint", type=str, default="data/sports_gpt.pt", help="path to model checkpoint")

    # 3. Sample options command (stretch goal)
    opt_parser = sub.add_parser("sample-options", help="generate multiple commentary options for a single prompt")
    opt_parser.add_argument("--prompt", "-p", type=str, default="Bottom of the ninth with bases loaded", help="game scenario prompt")
    opt_parser.add_argument("-n", "--num-samples", type=int, default=3, help="number of distinct commentary options to produce")
    opt_parser.add_argument("--tone", "-t", type=str, choices=["excited", "neutral", "dramatic"], default=None, help="tone or style")
    opt_parser.add_argument("--temperature", type=float, default=0.85, help="sampling temperature")
    opt_parser.add_argument("--checkpoint", type=str, default="data/sports_gpt.pt", help="path to model checkpoint")

    # 4. Train command
    train_parser = sub.add_parser("train", help="train the GPT model from scratch on sports text")
    train_parser.add_argument("--corpus", type=str, default="data/sports_corpus.txt", help="path to training corpus")
    train_parser.add_argument("--epochs", "-e", type=int, default=5, help="number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    train_parser.add_argument("--batch-size", "-b", type=int, default=16, help="training batch size")
    train_parser.add_argument("--save-path", "-s", type=str, default="data/sports_gpt.pt", help="output checkpoint path")

    # 5. Evaluate command (baseline comparison)
    eval_parser = sub.add_parser("evaluate", help="evaluate SportsGPT against uniform and unigram baselines")
    eval_parser.add_argument("--checkpoint", type=str, default="data/sports_gpt.pt", help="path to model checkpoint")

    # 6. Interactive command
    sub.add_parser("interactive", help="launch an interactive commentary generation shell")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")

    elif args.command == "generate":
        model, tokenizer = get_or_train_default_model(checkpoint_path=args.checkpoint)
        commentary = generate_commentary(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            tone=args.tone,
        )
        print("=== Sports Play-by-Play Commentary ===")
        print(f"Scenario Prompt : {args.prompt}")
        print(f"Selected Tone   : {args.tone}")
        print(f"Commentary      :\n{commentary}")

    elif args.command == "sample-options":
        model, tokenizer = get_or_train_default_model(checkpoint_path=args.checkpoint)
        print(f"=== Multi-Option Commentary for: '{args.prompt}' ===")
        for i in range(args.num_samples):
            opt = generate_commentary(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=40,
                temperature=args.temperature + (i * 0.05),
                tone=args.tone,
            )
            print(f"\n[Option {i + 1}]:\n{opt}")

    elif args.command == "train":
        print(f"Loading corpus from {args.corpus}...")
        corpus = CorpusLoader(args.corpus).load_text()
        tokenizer = BPETokenizer().train(corpus, max_vocab_size=150)
        token_ids = tokenizer.encode(corpus, add_special_tokens=False)
        dataset = SportsDataset(token_ids, seq_len=32, stride=8)

        print(f"Corpus tokens: {len(token_ids)} | Vocab size: {tokenizer.vocab_size} | Samples: {len(dataset)}")
        model = SportsGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=3,
            max_seq_len=128,
            dropout=0.1,
        )
        print(f"Training SportsGPT for {args.epochs} epochs with AdamW (lr={args.lr})...")
        losses = train_gpt(model, dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        save_checkpoint(args.save_path, model, tokenizer)
        print(f"Training complete! Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")
        print(f"Saved checkpoint to {args.save_path}")

    elif args.command == "evaluate":
        model, tokenizer = get_or_train_default_model(checkpoint_path=args.checkpoint)
        metrics = evaluate_baseline_vs_gpt(model, tokenizer)
        print("=== Model vs Baseline Comparison ===")
        print(f"1. Uniform Random Baseline  : Loss = {metrics['uniform_loss']:.4f} | Perplexity = {metrics['uniform_perplexity']:.2f}")
        print(f"2. Unigram Frequency Baseline: Loss = {metrics['unigram_loss']:.4f} | Perplexity = {metrics['unigram_perplexity']:.2f}")
        print(f"3. SportsGPT (Trained)      : Loss = {metrics['gpt_loss']:.4f} | Perplexity = {metrics['gpt_perplexity']:.2f}")

    elif args.command == "interactive":
        model, tokenizer = get_or_train_default_model()
        print("Sports Play-by-Play Commentary Assistant (type 'exit' or 'quit' to leave)")
        while True:
            try:
                prompt = input("\nEnter scenario prompt > ").strip()
                if not prompt or prompt.lower() in ("exit", "quit"):
                    break
                tone = input("Tone [excited/neutral/dramatic] (default: excited) > ").strip() or "excited"
                result = generate_commentary(model=model, tokenizer=tokenizer, prompt=prompt, tone=tone)
                print(f"\nAI Commentary:\n{result}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended. Good game!")


if __name__ == "__main__":
    main()
