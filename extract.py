"""Extract self-replicators at different fidelity levels for LLM pre-pretraining.

Mirrors the NCA pre-pretraining approach: generate structured synthetic sequences,
filter by replication fidelity, and output tokenized data for next-token prediction.

Usage:
    python extract.py --seeds 1 2 3 --num-epochs 7500
    python extract.py --seeds 1-10 --num-epochs 10000 --output-dir data/
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from main import (
    OPCODE_TOKENS,
    apply_background_mutation,
    build_neighborhood,
    run_epoch_pairs,
    run_tape,
    select_pairs,
    high_order_entropy,
)

OPCODE_SET = set(OPCODE_TOKENS.tolist())
BYTE_TO_BFF = {}
for b in range(256):
    if b in OPCODE_SET:
        BYTE_TO_BFF[b] = chr(b)
    else:
        BYTE_TO_BFF[b] = f"x{b:02x}"


def tape_to_tokens(tape: np.ndarray) -> list[str]:
    """Convert a raw tape to a list of tokens.

    Opcodes become their BFF character: < > { } + - . , [ ]
    Data bytes become hex tokens: x00 .. xff (excluding opcode bytes)
    """
    return [BYTE_TO_BFF[b] for b in tape]


def tape_gzip_complexity(tape: np.ndarray) -> float:
    """Gzip compression ratio as complexity proxy (like NCA paper).

    Returns compressed_size / raw_size. Lower = more compressible = simpler.
    """
    raw = tape.tobytes()
    compressed = gzip.compress(raw, compresslevel=6)
    return len(compressed) / len(raw)


def replication_score(tape: np.ndarray, rng: np.random.Generator, n_trials: int = 30) -> float:
    """Measure how well a tape copies itself onto a random partner."""
    scores = []
    for _ in range(n_trials):
        victim = rng.integers(0, 256, size=tape.shape[0], dtype=np.uint8)
        combined = np.concatenate([tape.copy(), victim])
        run_tape(combined)
        similarity = float(np.mean(combined[tape.shape[0]:] == tape))
        scores.append(similarity)
    return float(np.mean(scores))


def extract_unique_replicators(
    programs: np.ndarray, min_count: int = 2
) -> list[tuple[bytes, int]]:
    """Extract tapes that appear at least min_count times (evidence of replication)."""
    tape_size = programs.shape[-1]
    flat = programs.reshape(-1, tape_size)
    counts = Counter(map(bytes, flat))
    return [
        (tape_bytes, count)
        for tape_bytes, count in counts.most_common()
        if count >= min_count
    ]


def run_and_snapshot(
    seed: int,
    num_epochs: int,
    mutation_rate: float,
    grid_width: int,
    grid_height: int,
    tape_size: int,
    snapshot_epochs: list[int],
    min_replicator_count: int = 2,
) -> list[dict]:
    """Run spatial simulation, snapshot replicators at specified epochs."""
    rng = np.random.default_rng(seed)
    num_programs = grid_width * grid_height
    programs = rng.integers(
        0, 256, size=(grid_width, grid_height, tape_size), dtype=np.uint8
    )
    flat_programs = programs.reshape(num_programs, tape_size)

    neighbors, neighbor_counts = build_neighborhood(grid_width, grid_height)
    rows = np.arange(num_programs, dtype=np.int32)
    valid_rows = rows[neighbor_counts > 0]
    valid_neighbor_counts = neighbor_counts[valid_rows]
    all_rows_are_valid = valid_rows.size == num_programs
    pairs = np.empty((num_programs // 2, 2), dtype=np.int32)
    proposals = np.empty(num_programs, dtype=np.int32)
    taken = np.empty(num_programs, dtype=np.uint8)

    snapshot_set = set(snapshot_epochs)
    all_extractions: list[dict] = []
    score_rng = np.random.default_rng(seed + 999)

    pbar = tqdm(range(1, num_epochs + 1), desc=f"seed={seed} mut={mutation_rate:.1e}")
    for epoch in pbar:
        order = rng.permutation(num_programs).astype(np.int32, copy=False)
        if all_rows_are_valid:
            choices = rng.integers(0, neighbor_counts)
            proposals[:] = neighbors[rows, choices]
        else:
            proposals.fill(-1)
            choices = rng.integers(0, valid_neighbor_counts)
            proposals[valid_rows] = neighbors[valid_rows, choices]

        pair_count = select_pairs(order, proposals, pairs, taken)
        run_epoch_pairs(flat_programs, pairs, pair_count)
        apply_background_mutation(programs, mutation_rate, rng)

        if epoch in snapshot_set:
            replicators = extract_unique_replicators(flat_programs, min_replicator_count)
            hoe = high_order_entropy(programs)

            for tape_bytes, count in replicators:
                tape = np.frombuffer(tape_bytes, dtype=np.uint8).copy()
                tokens = tape_to_tokens(tape)
                complexity = tape_gzip_complexity(tape)
                opcode_count = sum(1 for b in tape if b in OPCODE_SET)
                rep_score = replication_score(tape, score_rng, n_trials=20)

                all_extractions.append({
                    "tokens": tokens,
                    "tape": list(int(b) for b in tape),
                    "seed": seed,
                    "epoch": epoch,
                    "mutation_rate": mutation_rate,
                    "count": count,
                    "frequency": count / num_programs,
                    "gzip_complexity": round(complexity, 4),
                    "opcode_ratio": round(opcode_count / tape_size, 4),
                    "replication_score": round(rep_score, 4),
                    "soup_hoe": round(hoe, 4),
                })

            n = len(replicators)
            pbar.set_postfix_str(f"epoch {epoch}: {n} reps, hoe={hoe:.3f}")

    return all_extractions


def assign_fidelity_tier(record: dict) -> str:
    """Bin replicators by replication fidelity.

    Measures how reliably a tape copies itself onto random partners.
    """
    rep = record.get("replication_score", 0)
    if rep < 0.01:
        return "noise"    # not a real replicator
    elif rep < 0.05:
        return "weak"     # partial replicator, copies a few bytes
    elif rep < 0.25:
        return "moderate" # copies significant structure
    else:
        return "strong"   # reliably overwrites partner with self


def parse_seed_range(s: str) -> list[int]:
    """Parse '1-10' or '1 2 3' into a list of ints."""
    if "-" in s and s.count("-") == 1 and not s.startswith("-"):
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split()]


def main():
    parser = argparse.ArgumentParser(description="Extract BFF self-replicators for LLM training")
    parser.add_argument("--seeds", type=str, default="1-3", help="Seeds: '1-10' or '1 2 3'")
    parser.add_argument("--num-epochs", type=int, default=7500)
    parser.add_argument("--grid-width", type=int, default=240)
    parser.add_argument("--grid-height", type=int, default=135)
    parser.add_argument("--tape-size", type=int, default=64)
    parser.add_argument(
        "--mutation-rates", type=float, nargs="+",
        default=[1e-4, 2.4e-4, 5e-4, 1e-3],
        help="Mutation rates to sweep",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=500,
        help="Snapshot replicators every N epochs",
    )
    parser.add_argument("--min-count", type=int, default=2, help="Min copies to count as replicator")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_epochs = list(range(args.snapshot_every, args.num_epochs + 1, args.snapshot_every))
    if args.num_epochs not in snapshot_epochs:
        snapshot_epochs.append(args.num_epochs)

    all_data: list[dict] = []
    for seed in seeds:
        for mut_rate in args.mutation_rates:
            print(f"\n=== seed={seed} mutation_rate={mut_rate:.1e} ===")
            extractions = run_and_snapshot(
                seed=seed,
                num_epochs=args.num_epochs,
                mutation_rate=mut_rate,
                grid_width=args.grid_width,
                grid_height=args.grid_height,
                tape_size=args.tape_size,
                snapshot_epochs=snapshot_epochs,
                min_replicator_count=args.min_count,
            )
            for e in extractions:
                e["fidelity_tier"] = assign_fidelity_tier(e)
            all_data.extend(extractions)
            print(f"  extracted {len(extractions)} replicator instances")

    # Write full dataset
    full_path = output_dir / "replicators_all.jsonl"
    with open(full_path, "w") as f:
        for d in all_data:
            f.write(json.dumps(d) + "\n")
    print(f"\nwrote {len(all_data)} total records -> {full_path}")

    # Write per-tier files for fidelity-filtered training
    tiers = ("noise", "weak", "moderate", "strong")
    tier_counts = {}
    for tier in tiers:
        tier_data = [d for d in all_data if d["fidelity_tier"] == tier]
        tier_counts[tier] = len(tier_data)
        if tier_data:
            tier_path = output_dir / f"replicators_{tier}.jsonl"
            with open(tier_path, "w") as f:
                for d in tier_data:
                    f.write(json.dumps(d) + "\n")

    # Write token-only files (one line = one tokenized tape, ready for LM training)
    # Exclude noise tier — those aren't real replicators
    for tier in ("weak", "moderate", "strong"):
        tier_data = [d for d in all_data if d["fidelity_tier"] == tier]
        if tier_data:
            tok_path = output_dir / f"tokens_{tier}.txt"
            with open(tok_path, "w") as f:
                for d in tier_data:
                    f.write(" ".join(d["tokens"]) + "\n")

    print(f"\nFidelity distribution:")
    for tier in tiers:
        print(f"  {tier:8s}: {tier_counts.get(tier, 0):6d}")

    # Deduplicated summary
    unique_tapes = {}
    for d in all_data:
        key = tuple(d["tape"])
        if key not in unique_tapes or d["replication_score"] > unique_tapes[key]["replication_score"]:
            unique_tapes[key] = d

    summary_path = output_dir / "replicators_unique.jsonl"
    with open(summary_path, "w") as f:
        for d in sorted(unique_tapes.values(), key=lambda x: -x["replication_score"]):
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(unique_tapes)} unique replicators -> {summary_path}")


if __name__ == "__main__":
    main()
