"""Visualize extracted replicators as colored 8x8 tiles, grouped by fidelity tier."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import NORMALIZE_LOOKUP, build_color_lut

COLOR_LUT = build_color_lut()
TIER_ORDER = ["strong", "moderate", "weak"]
TILE_PX = 32  # each cell in the 8x8 tape rendered at this many pixels
CELL_PX = TILE_PX // 8  # 4px per cell
MARGIN = 2


def tape_to_image(tape: list[int], scale: int = CELL_PX) -> Image.Image:
    """Render a 64-byte tape as an 8x8 colored image."""
    arr = np.array(tape, dtype=np.uint8)
    normalized = NORMALIZE_LOOKUP[arr].reshape(8, 8)
    colored = COLOR_LUT[normalized]  # (8, 8, 3)
    img = Image.fromarray(colored, mode="RGB")
    if scale > 1:
        img = img.resize((8 * scale, 8 * scale), Image.NEAREST)
    return img


def make_tier_strip(records: list[dict], max_per_tier: int = 50, scale: int = 6) -> Image.Image:
    """Render a horizontal strip of replicator tiles for one tier."""
    n = min(len(records), max_per_tier)
    if n == 0:
        return Image.new("RGB", (1, 1))
    tile_size = 8 * scale
    gap = 2
    width = n * (tile_size + gap) - gap
    img = Image.new("RGB", (width, tile_size), (30, 30, 30))
    for i, rec in enumerate(records[:n]):
        tile = tape_to_image(rec["tape"], scale=scale)
        img.paste(tile, (i * (tile_size + gap), 0))
    return img


def make_mosaic(
    records: list[dict],
    tier: str,
    cols: int = 20,
    scale: int = 6,
    sort_by: str = "replication_score",
) -> Image.Image:
    """Render a grid mosaic of replicator tiles."""
    sorted_recs = sorted(records, key=lambda r: -r.get(sort_by, 0))
    n = len(sorted_recs)
    rows = math.ceil(n / cols)
    tile_size = 8 * scale
    gap = 2
    width = cols * (tile_size + gap) - gap
    height = rows * (tile_size + gap) - gap
    img = Image.new("RGB", (width, height), (30, 30, 30))

    for i, rec in enumerate(sorted_recs):
        row, col = divmod(i, cols)
        tile = tape_to_image(rec["tape"], scale=scale)
        img.paste(tile, (col * (tile_size + gap), row * (tile_size + gap)))

    return img


def make_overview(data_path: str, output_dir: str, max_per_tier: int = 200) -> None:
    """Create per-tier mosaics and a combined overview."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tier_data = {}
    for tier in TIER_ORDER:
        tier_recs = [r for r in records if r.get("fidelity_tier") == tier]
        tier_recs.sort(key=lambda r: -r.get("replication_score", 0))
        tier_data[tier] = tier_recs[:max_per_tier]

    # Per-tier mosaics
    for tier, recs in tier_data.items():
        if not recs:
            continue
        mosaic = make_mosaic(recs, tier, cols=20, scale=6)
        path = output / f"replicators_{tier}.png"
        mosaic.save(path)
        print(f"  {tier:8s}: {len(recs):4d} replicators -> {path}")

    # Combined vertical layout with labels
    scale = 6
    tile_size = 8 * scale
    gap = 2
    cols = 20
    label_height = 24
    section_gap = 12

    sections = []
    total_height = 0
    max_width = 0

    for tier in TIER_ORDER:
        recs = tier_data[tier]
        if not recs:
            continue
        mosaic = make_mosaic(recs, tier, cols=cols, scale=scale)
        sections.append((tier, mosaic, len(recs)))
        total_height += label_height + mosaic.height + section_gap
        max_width = max(max_width, mosaic.width)

    if not sections:
        print("No replicators to visualize")
        return

    total_height -= section_gap  # no gap after last
    combined = Image.new("RGB", (max_width, total_height), (20, 20, 20))
    draw = ImageDraw.Draw(combined)

    try:
        font = ImageFont.truetype("consola.ttf", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

    y = 0
    for tier, mosaic, count in sections:
        thresholds = {"strong": 0.25, "moderate": 0.05, "weak": 0.01}
        label = f"{tier.upper()} (n={count}, rep_score>={thresholds[tier]})"
        draw.text((4, y + 4), label, fill=(200, 200, 200), font=font)
        y += label_height
        combined.paste(mosaic, (0, y))
        y += mosaic.height + section_gap

    combined_path = output / "replicators_overview.png"
    combined.save(combined_path)
    print(f"\n  overview -> {combined_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/replicators_all.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-per-tier", type=int, default=200)
    args = parser.parse_args()

    make_overview(args.data, args.output_dir, args.max_per_tier)
