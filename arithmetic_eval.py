"""
Evaluate the NextTerm model on synthetic polynomial sequences (arithmetic through
quartic) with varying amounts of prior context.

For each prior term count (default: 2-25 for linear, 3-25 for quadratic,
4-25 for cubic, 5-25 for quartic), we sample sequences from
ax^4 + bx^3 + cx^2 + dx + e with coefficient ranges:
    e in [0, 99999]
    d in [0, 9999]
    c in [0, 999]
    b in [0, 99]
    a in [0, 9]

Higher-order coefficients are set to 0 for lower-degree tests (e.g., arithmetic
sequences only use d and e).
"""

import argparse
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.tuner.utils import print_trainable_parameters
from tqdm import tqdm

# Step 335000
MODEL_NAME = "NextTerm-47M"
MAX_NEW_TOKENS = 20
DEFAULT_MAX_TERMS = 25
SAMPLES_PER_K = 200
RANDOM_SEED = 0

COEFF_MAX = {"a": 9, "b": 99, "c": 999, "d": 9999, "e": 99999}


@dataclass(frozen=True)
class PolynomialTask:
    name: str
    degree: int
    min_terms: int


TASKS = (
    PolynomialTask("arithmetic", 1, 2),
    PolynomialTask("quadratic", 2, 3),
    PolynomialTask("cubic", 3, 4),
    PolynomialTask("quartic", 4, 5),
)


def parse_generated(text: str):
    if "," in text:
        text = text.split(",")[0]
    try:
        return int(text)
    except ValueError:
        print(f"Could not parse generated text: {text!r}")
        return None


def polynomial_value(coeffs: Tuple[int, int, int, int, int], x: int) -> int:
    a, b, c, d, e = coeffs
    return (
        a * (x**4)
        + b * (x**3)
        + c * (x**2)
        + d * x
        + e
    )


def sample_coefficients(
    degree: int, rng: random.Random
) -> Tuple[int, int, int, int, int]:
    a = rng.randint(0, COEFF_MAX["a"]) if degree >= 4 else 0
    b = rng.randint(0, COEFF_MAX["b"]) if degree >= 3 else 0
    c = rng.randint(0, COEFF_MAX["c"]) if degree >= 2 else 0
    d = rng.randint(0, COEFF_MAX["d"]) if degree >= 1 else 0
    e = rng.randint(0, COEFF_MAX["e"])
    return a, b, c, d, e


def generate_polynomial_sequences(
    task: PolynomialTask, max_terms: int, samples_per_k: int, rng: random.Random
) -> List[Dict[str, int | List[int] | str]]:
    sequences = []
    for k in range(task.min_terms, max_terms + 1):
        for _ in range(samples_per_k):
            coeffs = sample_coefficients(task.degree, rng)
            seq = [polynomial_value(coeffs, i) for i in range(k + 1)]
            sequences.append(
                {"k": k, "prompt": seq[:-1], "answer": seq[-1], "task": task.name}
            )
    return sequences


def build_prompts(
    sequences: List[Dict[str, int | List[int] | str]], tokenizer
) -> Tuple[List[List[int]], List[Dict[str, int | List[int] | str]]]:
    prompts = []
    metadata = []
    for record in sequences:
        prompt_text = ",".join(str(x) for x in record["prompt"]) + ","
        prompts.append(tokenizer.encode(prompt_text))
        metadata.append(record)
    return prompts, metadata


def evaluate_task(
    task: PolynomialTask,
    args,
    model,
    tokenizer,
    sep_token: int,
) -> Tuple[Dict[int, Dict[str, int]], List[Dict[str, object]]]:
    rng = random.Random(args.seed + task.degree)
    sequences = generate_polynomial_sequences(
        task, args.max_terms, args.samples_per_k, rng
    )
    if not sequences:
        print(
            f"Skipping {task.name}: min_terms {task.min_terms} exceeds max_terms "
            f"{args.max_terms}"
        )
        return {}, []

    print(
        f"\nEvaluating {task.name} sequences (degree {task.degree}) with "
        f"{len(sequences)} samples ({args.samples_per_k} per prior-term count)"
    )

    prompts, metadata = build_prompts(sequences, tokenizer)

    gen = BatchGenerator(model, stop_tokens=[sep_token, tokenizer.eos_token_id])
    uids = gen.insert(prompts, args.max_new_tokens)

    meta_by_uid = dict(zip(uids, metadata))
    results = {uid: [] for uid in uids}
    finished = set()
    with tqdm(total=len(uids), desc=f"{task.name.title()}") as pbar:
        while responses := gen.next():
            for r in responses:
                results[r.uid].append(r.token)
                if r.finish_reason == "stop" and r.uid not in finished:
                    finished.add(r.uid)
                    pbar.update(1)

    stats = {
        k: {"correct": 0, "parsed": 0, "total": 0}
        for k in range(task.min_terms, args.max_terms + 1)
    }

    for uid in uids:
        tokens = results[uid]
        text = tokenizer.decode(tokens[:-1]) if tokens else ""
        prediction = parse_generated(text)

        meta = meta_by_uid[uid]
        k = meta["k"]
        answer = meta["answer"]

        stats[k]["total"] += 1
        if prediction is not None:
            stats[k]["parsed"] += 1
            if prediction == answer:
                stats[k]["correct"] += 1

    print("Accuracy by prior term count:")
    csv_rows: List[Dict[str, object]] = []
    for k in range(task.min_terms, args.max_terms + 1):
        s = stats[k]
        accuracy = s["correct"] / s["total"] if s["total"] else 0.0
        print(
            f"{k} terms: parsed {s['parsed']}/{s['total']} "
            f"accuracy {s['correct']}/{s['total']} = {accuracy:.4f}"
        )
        csv_rows.append(
            {
                "task": task.name,
                "degree": task.degree,
                "k": k,
                "parsed": s["parsed"],
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": accuracy,
            }
        )

    overall_total = sum(s["total"] for s in stats.values())
    overall_parsed = sum(s["parsed"] for s in stats.values())
    overall_correct = sum(s["correct"] for s in stats.values())
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0
    print(
        f"Overall {task.name}: parsed {overall_parsed}/{overall_total} "
        f"accuracy {overall_correct}/{overall_total} = {overall_accuracy:.4f}"
    )

    return stats, csv_rows


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        print("No results to write to CSV.")
        return

    fieldnames = ["task", "degree", "k", "parsed", "total", "correct", "accuracy"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote results to {path}")


def evaluate(args):
    random.seed(args.seed)

    model, tokenizer = load(args.model_name)

    print_trainable_parameters(model)

    sep_token = tokenizer.encode("1,")[-1]
    all_rows: List[Dict[str, object]] = []
    for task in TASKS:
        _, rows = evaluate_task(task, args, model, tokenizer, sep_token)
        all_rows.extend(rows)

    if args.csv_path:
        write_csv(all_rows, args.csv_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max_terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument("--samples_per_k", type=int, default=SAMPLES_PER_K)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--csv_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
