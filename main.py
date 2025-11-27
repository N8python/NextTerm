import json
import os
import random
import time

os.environ.setdefault("WANDB_SILENT", "true")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import wandb
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import ModelArgs, Model
from tokenize_utils import tokenize_sequence, detokenize_sequence_str, VOCAB_SIZE, PAD
from tqdm import tqdm

BATCH_SIZE = 64
SAVE_INTERVAL = 2500
VAL_INTERVAL = 1000
VAL_BATCHES = 100
TRAIN_PATH = "oeis_train_shuffled_enhanced.jsonl"

model_args = ModelArgs(
    model_type="qwen3",
    hidden_size=512,
    num_hidden_layers=12,
    intermediate_size=2048,
    num_attention_heads=8,
    rms_norm_eps=1e-6,
    vocab_size=VOCAB_SIZE,
    tie_word_embeddings=False,
    num_key_value_heads=4,
    max_position_embeddings=2048,
    rope_theta=10000,
    head_dim=512 // 8,
)

model = Model(model_args)

with open("oeis_val.jsonl", "r") as f:
    lines = f.readlines()
    val_data = [json.loads(line) for line in lines]

# Data comes pre-shuffled, one epoch over it


def count_lines(path):
    with open(path, "r") as f:
        return sum(1 for _ in f)


def stream_train_batches(path, batch_size):
    """Yield full batches from the training jsonl without keeping it all in memory."""
    batch = []
    with open(path, "r") as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []


def loss_fn(model, toks):
    targets = toks[:, 1:]
    inputs = toks[:, :-1]
    logits = model(inputs)
    loss = nn.losses.cross_entropy(logits, targets) * (targets != PAD)
    tok_count = (targets != PAD).sum()
    loss = loss.sum() / tok_count
    return loss, tok_count


def tokenize_and_pad_batch(batch):
    input_ids = []
    for entry in batch:
        seq = entry["seq"]
        tokens = tokenize_sequence(seq)
        input_ids.append(tokens)
    max_length = max(len(ids) for ids in input_ids)
    # Pad sequences to max length
    input_ids_padded = []
    for ids in input_ids:
        padded = ids + [PAD] * (max_length - len(ids))
        input_ids_padded.append(padded)
    input_ids_padded = mx.array(input_ids_padded, dtype=mx.int32)
    return input_ids_padded


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
LR = 1e-4
MUON_LR = 1e-2
train_example_count = count_lines(TRAIN_PATH)
TOTAL_STEPS = train_example_count // BATCH_SIZE
WARMUP = 10
linear = optim.linear_schedule(0, LR, steps=WARMUP)
cosine = optim.cosine_decay(LR, TOTAL_STEPS - WARMUP, LR * 0.1)
lr_schedule = optim.join_schedules([linear, cosine], [WARMUP])
muon_linear = optim.linear_schedule(0, MUON_LR, steps=WARMUP)
muon_cosine = optim.cosine_decay(MUON_LR, TOTAL_STEPS - WARMUP, MUON_LR * 0.1)
muon_schedule = optim.join_schedules([muon_linear, muon_cosine], [WARMUP])


def criteria(path, weight):
    is_good_for_adamw = weight.ndim != 2 or "embed" in path or "lm_head" in path
    return is_good_for_adamw


optimizer = optim.MultiOptimizer(
    [optim.AdamW(learning_rate=lr_schedule), optim.Muon(learning_rate=muon_schedule, weight_decay=0.0)],
    [criteria],
)

wandb.init(
    project="oeis-bigger",
    mode=os.environ.get("WANDB_MODE", "online"),
    config={
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "muon_lr": MUON_LR,
        "total_steps": TOTAL_STEPS,
        "warmup": WARMUP,
        "save_interval": SAVE_INTERVAL,
        "val_interval": VAL_INTERVAL,
        "val_batches": VAL_BATCHES,
    },
)

os.makedirs("checkpoints", exist_ok=True)

pbar = tqdm(stream_train_batches(TRAIN_PATH, BATCH_SIZE), total=TOTAL_STEPS)
for step, batch in enumerate(pbar):
    step_start = time.time()
    mx.clear_cache()

    input_ids_padded = tokenize_and_pad_batch(batch)

    (loss, tok_count), grads = loss_and_grad_fn(model, input_ids_padded)
    optimizer.update(model, grads)
    elapsed = time.time() - step_start
    tokens_per_sec = round(tok_count.item() / elapsed) if elapsed > 0 else 0
    postfix = {"train_loss": f"{loss.item():.4f}", "tps": f"{tokens_per_sec/1000:.1f}K"}
    wandb.log(
        {"train/loss": loss.item(), "train/tokens": tok_count.item(), "step": step, "train/tokens_per_sec": tokens_per_sec}
    )
    if step % VAL_INTERVAL == 0:
        # Run validation
        random.shuffle(val_data)
        val_loss = 0.0
        val_tok_count = 0
        for j in range(0, VAL_BATCHES * BATCH_SIZE, BATCH_SIZE):
            mx.clear_cache()
            batch = val_data[j : j + BATCH_SIZE]

            input_ids_padded = tokenize_and_pad_batch(batch)

            loss, tok_count = loss_fn(model, input_ids_padded)
            val_loss += loss.item() * tok_count.item()
            val_tok_count += tok_count.item()
        val_loss /= val_tok_count
        wandb.log({"val/loss": val_loss, "val/tokens": val_tok_count, "step": step})
        postfix["val_loss"] = f"{val_loss:.4f}"
    if step % SAVE_INTERVAL == 0:
        # Save checkpoint
        checkpoint_path = f"checkpoints/model_step_{step}.ckpt"
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(checkpoint_path, adapter_weights)
        wandb.log({"checkpoint/step": step, "checkpoint/path": checkpoint_path})
        postfix["ckpt"] = step
    pbar.set_postfix(postfix)
# Final save
checkpoint_path = f"checkpoints/model_final.ckpt"
adapter_weights = dict(tree_flatten(model.trainable_parameters()))
mx.save_safetensors(checkpoint_path, adapter_weights)
wandb.log({"checkpoint/final_path": checkpoint_path})
