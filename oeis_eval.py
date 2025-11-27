"""
Evaluate the NextTerm model on oeis_val_decontam.jsonl by predicting the last
term of each sequence.
"""

import json
from pathlib import Path

from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from tqdm import tqdm

DATA_PATH = Path("oeis_val_decontam.jsonl")
MODEL_NAME = "Qwen3-14B-Base"
#MODEL_NAME = "NextTerm-Alpha-50M"
MAX_NEW_TOKENS = 20


def parse_generated(text: str):
    if "," in text:
        text = text.split(",")[0]
    try:
        return int(text)
    except ValueError:
        print(f"Could not parse generated text: {text!r}")
        return None


def load_sequences(path: Path):
    sequences = []
    answers = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            seq = record.get("seq", [])
            if len(seq) < 2:
                continue
            sequences.append(seq[:-1])
            answers.append(seq[-1])
    return sequences, answers


def main():
    sequences, answers = load_sequences(DATA_PATH)
    print(f"Loaded {len(answers)} sequences from {DATA_PATH}")

    model, tokenizer = load(MODEL_NAME)

    sep_token = tokenizer.encode("1,")[-1]
    prompts = [",".join(str(x) for x in seq) + "," for seq in sequences]
    prompts = [tokenizer.encode(p) for p in prompts]

    gen = BatchGenerator(model, stop_tokens=[sep_token, tokenizer.eos_token_id])
    uids = gen.insert(prompts, MAX_NEW_TOKENS)

    results = {uid: [] for uid in uids}
    finished = set()
    with tqdm(total=len(uids), desc="Generating") as pbar:
        while responses := gen.next():
            for r in responses:
                results[r.uid].append(r.token)
                if r.finish_reason == "stop" and r.uid not in finished:
                    finished.add(r.uid)
                    pbar.update(1)

    texts = [tokenizer.decode(results[uid][:-1]) for uid in uids]
    predictions = [parse_generated(text) for text in texts]

    correct = sum(1 for p, a in zip(predictions, answers) if p == a)
    parsed = sum(1 for p in predictions if p is not None)
    total = len(answers)

    print(f"Parsed predictions: {parsed}/{total}")
    print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    main()
