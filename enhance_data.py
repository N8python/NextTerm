import random
import json
from tokenize_utils import tokenize_sequence

def mult_seq(seq):
    mult = random.randint(2, 5)
    return [x * mult for x in seq]

def negate_sequence(seq):
    return [-x for x in seq]

def alternating_negation(seq):
    return [x if i % 2 == 0 else -x for i, x in enumerate(seq)]

def shift_sequence(seq):
    shift = random.randint(-10, 10)
    return [x + shift for x in seq]

def swap_pairs(seq):
    swapped = seq[:]
    for i in range(0, len(swapped) - 1, 2):
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    return swapped

def subsample_sequence(seq):
    """Take every nth element"""
    if len(seq) < 6:
        return seq
    step = random.randint(2, 3)
    result = seq[::step]
    return result if len(result) >= 3 else seq

def differences(seq):
    """First differences: seq[i+1] - seq[i]"""
    if len(seq) < 4:
        return seq
    return [seq[i+1] - seq[i] for i in range(len(seq) - 1)]

def partial_sums(seq):
    """Cumulative sum"""
    result = []
    total = 0
    for x in seq:
        total += x
        result.append(total)
    return result

def modular_reduction(seq):
    """Reduce all elements mod some value"""
    mod = random.randint(2, 10)
    return [x % mod for x in seq]

def truncate_sequence(seq):
    """Take a random contiguous subsequence"""
    if len(seq) < 6:
        return seq
    start = random.randint(0, len(seq) // 3)
    end = random.randint(2 * len(seq) // 3, len(seq))
    result = seq[start:end]
    return result if len(result) >= 3 else seq

# Length-preserving transforms (safe to chain multiple times)
length_preserving = [mult_seq, negate_sequence, alternating_negation, shift_sequence, swap_pairs, partial_sums, modular_reduction]

# Length-reducing transforms (apply at most once)
length_reducing = [subsample_sequence, differences, truncate_sequence]

def apply_transforms(seq):
    """Apply random transforms, being careful with length-reducing ones"""
    transformed = seq[:]
    
    # Maybe apply one length-reducing transform first
    if random.random() < 0.3 and len(transformed) >= 6:
        transform = random.choice(length_reducing)
        transformed = transform(transformed)
    
    # Apply 0-2 length-preserving transforms
    num_transforms = random.randint(0, 2)
    for _ in range(num_transforms):
        transform = random.choice(length_preserving)
        transformed = transform(transformed)
    
    return transformed

def main():
    target_tokens = 5_000_000_000
    current_tokens = 0
    min_seq_length = 3
    
    # First pass: read original data and count tokens
    print("Reading original data...")
    data = []
    with open("oeis_train.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    
    print(f"Loaded {len(data)} original sequences")
    
    # Stream output
    with open("oeis_train_uber.jsonl", "w") as out_f:
        # Write original sequences first
        print("Writing original sequences...")
        for entry in data:
            out_f.write(json.dumps(entry) + "\n")
            current_tokens += len(tokenize_sequence(entry["seq"]))
        
        print(f"Original tokens: {current_tokens:,}")
        
        # Generate augmented sequences
        added = 0
        while current_tokens < target_tokens:
            entry = random.choice(data)
            seq = entry["seq"]
            
            if len(seq) < min_seq_length:
                continue
            
            transformed_seq = apply_transforms(seq)
            
            # Skip if too short or unchanged
            if len(transformed_seq) < min_seq_length:
                continue
            
            new_entry = {
                "seq": transformed_seq,
                "id": entry["id"] + "_aug"
            }
            
            out_f.write(json.dumps(new_entry) + "\n")
            current_tokens += len(tokenize_sequence(transformed_seq))
            added += 1
            
            if added % 10000 == 0:
                progress = current_tokens / target_tokens
                print(f"Added {added:,} augmented | tokens: {current_tokens:,} | progress: {progress:.4%}")
    
    print(f"Done! Final token count: {current_tokens:,}")
    print(f"Total augmented sequences added: {added:,}")

if __name__ == "__main__":
    main()