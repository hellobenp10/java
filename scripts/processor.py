#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic denomination block processor

- Detects canonical block per family (credit, $1, $2) dynamically
- Extracts semantic template and base values from canonical (or baseline hints)
- Scales and overlays values into other blocks based on computed scale factors
- Detects and preserves cash overlays (e.g., "$600") and notes/FG grids
- Populates overlay bands 7–12, 20–25, 32–37, 45–50 relative to each 12-row slice
- Processes CSVs from inputs/ to output/

Usage:
  python scripts/processor.py  # expects inputs/*.csv

The code avoids hard-coding absolute row numbers; overlay ranges are generated per 12-row segments.
"""

import os
import csv
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ==============================
# Configuration and constants
# ==============================

# Width of one denomination block (columns per block)
BLOCK_WIDTH = 6

# Known semantic labels that can appear inside blocks
SEMANTIC_LABELS = {
    "Grand", "Super", "Major", "Max", "Mor", "Mini",
    "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
    "FG", "ITS ALIVE"
}

# Symbols considered base anchors (their immediate right neighbor should be numeric in a well-formed block)
BASE_LABELS = {"Max", "Mor", "Mini", "g", "h", "i", "d", "e", "f", "a", "b", "c"}

# Multipliers allowed for derived detection
DERIVED_MULTIPLIERS = list(range(1, 11))  # 1×..10×

# Preferred canonical denomination per group (used if present; otherwise we fall back dynamically)
# credit_low covers 1CT/2CT, credit_hi covers 5CT/10CT
PREFERRED_GROUP_CANONICAL = {
    "credit_low": ("1CT", 0.60),
    "credit_hi": ("5CT", 1.50),
    "$1": ("$1", 10.0),
    "$2": ("$2", 20.0),
}

# Families and simple detection helpers
CREDIT_TOKENS = {"1CT", "2CT", "5CT", "10CT"}
DOLLAR_TOKENS = {"$1", "$2"}

# Explicit family map and scale factors (used as hints and for validation/fallback)
FAMILY_MAP: Dict[str, str] = {
    # credit family
    "1CT $0.60": "credit", "1CT $1.20": "credit", "1CT $1.80": "credit",
    "1CT $3.00": "credit", "1CT $6.00": "credit",
    "2CT $1.20": "credit", "2CT $2.40": "credit", "2CT $3.60": "credit",
    "2CT $6.00": "credit", "2CT $12.00": "credit",
    "5CT $1.50": "credit", "5CT $3.00": "credit", "5CT $4.50": "credit",
    "5CT $7.50": "credit", "5CT $15.00": "credit",
    "10CT $3.00": "credit", "10CT $6.00": "credit", "10CT $9.00": "credit",
    "10CT $15.00": "credit", "10CT $30.00": "credit",

    # $1 family
    "$1 $5": "$1", "$1 $10": "$1", "$1 $15": "$1", "$1 $20": "$1", "$1 $25": "$1",

    # $2 family
    "$2 $10": "$2", "$2 $20": "$2", "$2 $30": "$2", "$2 $40": "$2", "$2 $50": "$2",
}

SCALE_FACTORS: Dict[str, float] = {
    # credit family (relative to 1CT $0.60)
    "1CT $0.60": 1, "1CT $1.20": 2, "1CT $1.80": 3, "1CT $3.00": 5, "1CT $6.00": 10,
    "2CT $1.20": 2, "2CT $2.40": 4, "2CT $3.60": 6, "2CT $6.00": 10, "2CT $12.00": 20,
    "5CT $1.50": 2.5, "5CT $3.00": 5, "5CT $4.50": 7.5, "5CT $7.50": 12.5, "5CT $15.00": 25,
    "10CT $3.00": 5, "10CT $6.00": 10, "10CT $9.00": 15, "10CT $15.00": 25, "10CT $30.00": 50,

    # $1 family (relative to $1 $10)
    "$1 $10": 1, "$1 $5": 0.5, "$1 $15": 1.5, "$1 $20": 2, "$1 $25": 2.5,

    # $2 family (relative to $2 $20)
    "$2 $20": 1, "$2 $10": 0.5, "$2 $30": 1.5, "$2 $40": 2, "$2 $50": 2.5,
}

# Row → label mapping used for baseline construction and validation
BASE_ROW_LABELS: Dict[int, List[str]] = {
    3: ["Max", "Mor", "Mini"],
    4: ["g", "h", "i"],
    5: ["d", "e", "f"],
    6: ["a", "b", "c"],
}

# Optional baselines for specific canonical product keys.
# Values are floats (strip any $/comma in the source before adding here).
# For the credit family, row 3 entries are CASH amounts; rows 4–6 are CREDITS.
# For $ families, rows 3–6 are CASH amounts.
BASELINE_ROWS_BY_KEY: Dict[str, Dict[int, List[float]] ] = {
    # Credit family canonical example
    "1CT $0.60": {
        3: [30.0, 15.0, 10.0],
        4: [1000.0, 500.0, 300.0],
        5: [200.0, 150.0, 125.0],
        6: [100.0, 75.0, 60.0],
    },
    # Scaled from 1CT $0.60 by 5
    "1CT $3.00": {
        3: [150.0, 75.0, 50.0],
        4: [5000.0, 2500.0, 1500.0],
        5: [1000.0, 750.0, 625.0],
        6: [500.0, 375.0, 300.0],
    },
    # $2 family example (given for $2 $50.00)
    "$2 $50.00": {
        3: [80.0, 60.0, 50.0],
        4: [2500.0, 1250.0, 750.0],
        5: [750.0, 400.0, 250.0],
        6: [200.0, 150.0, 100.0],
    },
}

def _rows_scale(rows: Dict[int, List[float]], factor: float) -> Dict[int, List[float]]:
    return {r: [round(v * factor, 6) for v in vs] for r, vs in rows.items()}


def family_of_label_token(token: str) -> str:
    token = token.strip()
    if any(token.startswith(ct) for ct in CREDIT_TOKENS) or "CT" in token:
        return "credit"
    if token.startswith("$1"):
        return "$1"
    if token.startswith("$2"):
        return "$2"
    return "unknown"


def _family_of_key(full_key: str) -> str:
    tok = full_key.split(" ")[0] if full_key else ""
    return family_of_label_token(tok)


def get_baseline_rows_for_key(product_key: str) -> Optional[Dict[int, List[float]]]:
    """Return baseline rows for a product_key, scaling from a known baseline if needed."""
    if product_key in BASELINE_ROWS_BY_KEY:
        return BASELINE_ROWS_BY_KEY[product_key]

    fam = _family_of_key(product_key)
    if fam == "unknown":
        return None

    # Try to find any baseline in same family we can scale from using SCALE_FACTORS
    if product_key in SCALE_FACTORS:
        target_rel = SCALE_FACTORS[product_key]
        # Prefer a natural canonical per family
        preferred = {
            "credit": "1CT $0.60",
            "$1": "$1 $10",
            "$2": "$2 $20",
        }.get(fam)
        candidates = []
        for base_key, rows in BASELINE_ROWS_BY_KEY.items():
            if _family_of_key(base_key) != fam:
                continue
            if base_key in SCALE_FACTORS:
                candidates.append(base_key)
        # Try preferred first if present, else any candidate
        base_key = preferred if preferred in candidates else (candidates[0] if candidates else None)
        if base_key:
            base_rel = SCALE_FACTORS.get(base_key, 1.0)
            if base_rel:
                factor = target_rel / base_rel
                return _rows_scale(BASELINE_ROWS_BY_KEY[base_key], factor)
    return None


def build_base_from_baseline(product_key: str, denom_value: float) -> Optional[Dict[str, Dict[str, float]]]:
    rows = get_baseline_rows_for_key(product_key)
    if not rows:
        return None
    fam = _family_of_key(product_key)
    base_values: Dict[str, Dict[str, float]] = {}
    for row_idx, labels in BASE_ROW_LABELS.items():
        values = rows.get(row_idx)
        if not values or len(values) != 3:
            continue
        for i, label in enumerate(labels):
            v = float(values[i])
            if fam == "credit":
                if row_idx == 3:
                    # row 3 is cash; convert to credits using denom_value
                    credits = (v / denom_value) if denom_value else v
                    base_values[label] = {"credits": round(credits, 6), "cash": round(v, 6)}
                else:
                    # rows 4..6 are credits
                    base_values[label] = {"credits": round(v, 6), "cash": round(v * denom_value, 6)}
            else:
                # $ families treat all as cash amounts; store as credits numerically for multiplier logic
                base_values[label] = {"credits": round(v, 6), "cash": round(v, 6)}
    return base_values if base_values else None


# ==============================
# Data classes
# ==============================

@dataclass
class DenomBlock:
    key: str                     # e.g., "1CT $0.60"
    family: str                  # "credit" | "$1" | "$2" | "unknown"
    denom_value: float           # numeric denomination value (e.g., 0.60, 10.0, 20.0)
    start_col: int               # inclusive column index in the full DF
    end_col: int                 # inclusive column index in the full DF
    semantics: Dict[Tuple[int, int], str]
    base_values: Dict[str, Dict[str, float]]  # {label: {credits, cash}}
    derived: Dict[Tuple[int, int], Dict[str, Any]]


# ==============================
# Utilities
# ==============================

def clean_number(val: Any) -> Optional[float]:
    s = str(val).strip().replace(",", "")
    if not s:
        return None
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except Exception:
        return None


def is_numeric_str(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    s = s.replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    return bool(re.fullmatch(r"\d+(\.\d+)?", s))


def to_df(table: List[List[Any]]) -> pd.DataFrame:
    """Builds a pandas DataFrame from a ragged list-of-lists without treating any row as header."""
    if not table:
        return pd.DataFrame()
    width = max(len(r) for r in table)
    norm = [r + [""] * (width - len(r)) for r in table]
    return pd.DataFrame(norm)


def slice_blocks(df: pd.DataFrame, block_width: int = BLOCK_WIDTH) -> List[Tuple[int, int]]:
    """Return (start_col_idx, end_col_idx) for each block across the DF columns."""
    blocks = []
    ncols = df.shape[1]
    for start in range(0, ncols, block_width):
        end = min(start + block_width - 1, ncols - 1)
        if start <= end:
            blocks.append((start, end))
    return blocks


def parse_block_denom(df: pd.DataFrame, start: int, end: int) -> Tuple[str, float, Tuple[int, int]]:
    """Scan a block to find a denomination token (1CT/$1/$2/etc.) and its numeric value to the right."""
    for r in range(df.shape[0]):
        for c in range(start, end + 1):
            cell = str(df.iat[r, c]).strip()
            if not cell:
                continue
            if any(cell.startswith(t) for t in CREDIT_TOKENS | DOLLAR_TOKENS):
                nxt = str(df.iat[r, c + 1]).strip() if c + 1 <= end else ""
                val = clean_number(nxt) or 0.0
                return f"{cell} {nxt}".strip(), val, (r, c)
    return "", 0.0, (-1, -1)


def detect_semantics(df: pd.DataFrame, start: int, end: int) -> Dict[Tuple[int, int], str]:
    semantics: Dict[Tuple[int, int], str] = {}
    for r in range(df.shape[0]):
        for c in range(start, end + 1):
            val = str(df.iat[r, c]).strip()
            if val in SEMANTIC_LABELS or val in BASE_LABELS:
                semantics[(r, c)] = val
    return semantics


def extract_base_values(df: pd.DataFrame, start: int, end: int, denom_value: float) -> Dict[str, Dict[str, float]]:
    base: Dict[str, Dict[str, float]] = {}
    for r in range(df.shape[0]):
        for c in range(start, end):
            label = str(df.iat[r, c]).strip()
            if label in BASE_LABELS:
                right_raw = str(df.iat[r, c + 1]).strip()
                if is_numeric_str(right_raw):
                    credits = clean_number(right_raw)
                    if credits is None:
                        continue
                    base[label] = {"credits": float(credits), "cash": round(float(credits) * denom_value, 6)}
    return base


def detect_derived_values(
    df: pd.DataFrame,
    start: int,
    end: int,
    base_values: Dict[str, Dict[str, float]],
    denom_value: float,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    derived: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for r in range(df.shape[0]):
        for c in range(start, end + 1):
            raw = str(df.iat[r, c]).strip()
            if not raw:
                continue

            # Cash overlays (e.g., "$600")
            if raw.startswith("$") and is_numeric_str(raw):
                cash_val = clean_number(raw)
                if cash_val is not None:
                    derived[(r, c)] = {"credits": None, "cash": float(cash_val), "derived_from": "overlay"}
                continue

            # Numeric credits
            if not is_numeric_str(raw):
                continue
            val = float(clean_number(raw) or 0.0)

            matched = False
            for sym, bv in base_values.items():
                base_cr = float(bv["credits"]) if bv and "credits" in bv else None
                if base_cr is None or base_cr == 0.0:
                    continue
                for m in DERIVED_MULTIPLIERS:
                    if np.isclose(val, base_cr * m, rtol=1e-6, atol=1e-6):
                        derived[(r, c)] = {
                            "credits": val,
                            "cash": round(val * denom_value, 6),
                            "derived_from": f"{m}×{sym}" if m != 1 else sym,
                        }
                        matched = True
                        break
                if matched:
                    break
    return derived


# ==============================
# Canonical base selection per family
# ==============================

def choose_canonical_block(blocks: List["DenomBlock"]) -> Optional["DenomBlock"]:
    if not blocks:
        return None

    family = blocks[0].family

    # Split credit into two subgroups: low (1CT/2CT) vs high (5CT/10CT)
    if family == "credit":
        low = [b for b in blocks if any(b.key.startswith(tok) for tok in ("1CT", "2CT"))]
        hi = [b for b in blocks if any(b.key.startswith(tok) for tok in ("5CT", "10CT"))]
        # Prefer low group if present, else high group
        group_blocks = low if low else hi
        group_key = "credit_low" if low else "credit_hi"
        pref_token, pref_value = PREFERRED_GROUP_CANONICAL.get(group_key, (None, None))
    else:
        # $1 / $2 families use direct preferences
        pref_token, pref_value = PREFERRED_GROUP_CANONICAL.get(family, (None, None))

    # 1) Prefer exact token + closest to preferred denom value
    if pref_token is not None:
        candidates = [b for b in blocks if b.key.startswith(pref_token)]
        if candidates:
            # Choose the one whose denom_value is closest to preferred
            if pref_value is not None:
                best = min(candidates, key=lambda b: abs(b.denom_value - pref_value))
                return best
            # else arbitrary among candidates
            return candidates[0]

    # 2) Otherwise choose the one with the smallest denom_value (good default for credits)
    try:
        return min(blocks, key=lambda b: (b.denom_value if b.denom_value > 0 else float("inf")))
    except ValueError:
        return blocks[0]


# ==============================
# Overlay band mapping (dynamic, not hard-coded table heights)
# ==============================

def overlay_pairs(total_rows: int) -> List[Tuple[range, range]]:
    """
    Build overlay (base_rows -> overlay_rows) pairs dynamically per 12-row segment:
    For each segment of 12 rows, rows [offset+0..offset+5] overlay into [offset+6..offset+11].
    This implements the provided mapping without hard-coding absolute indices:
      7–12 overlay 1–6, 20–25 overlay 14–19, 32–37 overlay 26–31, 45–50 overlay 39–44, ...
    """
    pairs: List[Tuple[range, range]] = []
    for seg_start in range(0, total_rows, 12):
        base = range(seg_start + 0, min(seg_start + 6, total_rows))
        over = range(seg_start + 6, min(seg_start + 12, total_rows))
        if len(base) == 6 and len(over) == 6:
            pairs.append((base, over))
    return pairs


# ==============================
# Block extraction and scaling
# ==============================

def extract_blocks(df: pd.DataFrame) -> List[DenomBlock]:
    blocks: List[DenomBlock] = []
    for start, end in slice_blocks(df):
        key, denom, _ = parse_block_denom(df, start, end)
        fam = family_of_label_token(key.split(" ")[0]) if key else "unknown"
        semantics = detect_semantics(df, start, end)
        base_vals = extract_base_values(df, start, end, denom)
        # If table is missing bases, attempt to build from baselines/scales
        if not base_vals and key:
            baseline_built = build_base_from_baseline(key, denom)
            if baseline_built:
                base_vals = baseline_built
        derived_vals = detect_derived_values(df, start, end, base_vals, denom)
        blocks.append(
            DenomBlock(
                key=key or f"unknown_{start}_{end}",
                family=fam,
                denom_value=float(denom or 0.0),
                start_col=start,
                end_col=end,
                semantics=semantics,
                base_values=base_vals,
                derived=derived_vals,
            )
        )
    return blocks


def scale_from_canonical(blocks: List[DenomBlock], df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping keyed by block.key:
      - canonical_base key
      - scale_factor (relative to canonical denom_value)
      - semantic_template (positions from canonical)
      - regenerated base_values (scaled credits + recomputed cash)
      - derived (detected from values present in each block using regenerated bases)
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    # Group by family
    fam_to_blocks: Dict[str, List[DenomBlock]] = {}
    for b in blocks:
        fam_to_blocks.setdefault(b.family, []).append(b)

    for fam, fam_blocks in fam_to_blocks.items():
        # Find canonical for the family
        canonical = choose_canonical_block(fam_blocks)
        if not canonical:
            continue

        c_val = canonical.denom_value if canonical.denom_value else 1.0
        c_sem = canonical.semantics
        c_base = canonical.base_values if canonical.base_values else build_base_from_baseline(canonical.key, canonical.denom_value) or {}

        # If canonical block lacks bases (e.g., empty), keep empty; we'll just annotate semantics
        for b in fam_blocks:
            # Use explicit SCALE_FACTORS if keys are recognized; else fall back to denom ratio
            if canonical.key in SCALE_FACTORS and b.key in SCALE_FACTORS:
                # scale from canonical by dividing their relative factors
                # normalize canonical factor to 1 if not present
                canon_rel = SCALE_FACTORS.get(canonical.key, 1.0)
                blk_rel = SCALE_FACTORS.get(b.key, b.denom_value / c_val if c_val else 1.0)
                scale = float(blk_rel) / float(canon_rel) if canon_rel else 1.0
            else:
                scale = (b.denom_value / c_val) if c_val else 1.0

            # Regenerate/scale base credits from canonical base_values
            regen_base: Dict[str, Dict[str, float]] = {}
            if c_base:
                for sym, vals in c_base.items():
                    cr = float(vals.get("credits", 0.0))
                    sc_cr = round(cr * scale, 6)
                    regen_base[sym] = {
                        "credits": sc_cr,
                        "cash": round(sc_cr * b.denom_value, 6),
                    }

            # Detect derived inside this block using regenerated bases
            derived_vals = detect_derived_values(
                df, b.start_col, b.end_col, regen_base if regen_base else b.base_values, b.denom_value
            )

            mapping[b.key] = {
                "family": fam,
                "denom_value": b.denom_value,
                "canonical_base": canonical.key,
                "scale_factor": round(scale, 6),
                "semantic_template": c_sem if c_sem else b.semantics,
                "base_values": regen_base if regen_base else b.base_values,
                "derived": derived_vals if derived_vals else b.derived,
            }

    return mapping


# ==============================
# In-place update (labels + numbers) with overlays handled gracefully
# ==============================

def write_block_labels_and_numbers(
    df: pd.DataFrame,
    block: DenomBlock,
    info: Dict[str, Any],
) -> None:
    start, end = block.start_col, block.end_col

    # 1) Ensure semantic labels from canonical template are present in this block
    for (r, c), label in info.get("semantic_template", {}).items():
        if start <= c <= end and 0 <= r < df.shape[0]:
            # Only write label if cell is empty-ish (avoid clobbering explicit content like notes)
            cell = str(df.iat[r, c]).strip()
            if cell in {"", "·", "nan"} or cell not in SEMANTIC_LABELS | BASE_LABELS:
                df.iat[r, c] = label

    # 2) For each base label found in this block, write scaled credits to the immediate right cell
    base_vals = info.get("base_values", {})
    if base_vals:
        for r in range(df.shape[0]):
            for c in range(start, end):
                label = str(df.iat[r, c]).strip()
                if label in base_vals:
                    if c + 1 <= end:
                        df.iat[r, c + 1] = base_vals[label]["credits"]

    # 3) Overlays: for each 12-row segment, replicate numbers into overlay rows when a clean multiplier is present
    #    without clobbering explicit cash overlays or notes content. This keeps it dynamic.
    #    This also covers feature grids (FG) rows by preserving any existing non-empty content.
    for base_rng, over_rng in overlay_pairs(df.shape[0]):
        for i, r_base in enumerate(base_rng):
            if r_base >= df.shape[0]:
                continue
            r_over = over_rng[i] if i < len(over_rng) else None
            if r_over is None or r_over >= df.shape[0]:
                continue

            # For each column in this block, if the overlay cell looks empty-ish and
            # the base row has a semantic label + credit to the right, we can propagate
            for c in range(start, end):
                overlay_cell = str(df.iat[r_over, c]).strip()
                if overlay_cell not in {"", "·", "nan"}:
                    continue  # respect existing overlay text (FG, notes, explicit cash etc.)

                label_here = str(df.iat[r_base, c]).strip()
                # If overlay column cell is where a multiplier lives (x2..x10), try to place scaled value at right in overlay row
                if label_here.startswith("x") and re.fullmatch(r"x(\d+)", label_here):
                    try:
                        mult = int(label_here[1:])
                    except ValueError:
                        mult = None
                    if mult is None or mult <= 0:
                        continue

                    # Find nearest base label to the left in the same base row and use its credits
                    nearest_label_col: Optional[int] = None
                    nearest_label: Optional[str] = None
                    for lc in range(c - 1, start - 1, -1):
                        lbl = str(df.iat[r_base, lc]).strip()
                        if lbl in base_vals:
                            nearest_label_col = lc
                            nearest_label = lbl
                            break

                    if nearest_label and (c + 1) <= end:
                        base_cr = base_vals[nearest_label]["credits"]
                        df.iat[r_over, c] = label_here  # keep the xN marker in overlay row
                        df.iat[r_over, c + 1] = round(float(base_cr) * mult, 6)


# ==============================
# End-to-end processing
# ==============================

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Extract all blocks first
    blocks = extract_blocks(df)

    # Build scaling info from per-family canonical selection
    info_by_key = scale_from_canonical(blocks, df)

    # Write updates block-by-block (in input order to preserve left→right dependencies if any)
    for b in blocks:
        info = info_by_key.get(b.key)
        if not info:
            # No canonical info; still try to write existing bases/semantics for this block only
            info = {
                "semantic_template": b.semantics,
                "base_values": b.base_values,
            }
        write_block_labels_and_numbers(df, b, info)

    return df


# ==============================
# CSV I/O
# ==============================

def load_csv(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def save_csv(path: str, df: pd.DataFrame) -> None:
    # Save with no header and no index to preserve original shape-like output
    df.to_csv(path, index=False, header=False)


# ==============================
# CLI
# ==============================

def main(inputs_dir: str = "inputs", output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(inputs_dir):
        os.makedirs(inputs_dir, exist_ok=True)
        print("Created inputs/ directory. Place CSV files inside and rerun.")
        return

    any_found = False
    for name in sorted(os.listdir(inputs_dir)):
        if not name.lower().endswith(".csv"):
            continue
        in_path = os.path.join(inputs_dir, name)
        try:
            table = load_csv(in_path)
            df = to_df(table)
            updated = process_dataframe(df)
            out_path = os.path.join(output_dir, f"processed_{name}")
            save_csv(out_path, updated)
            print(f"Processed {name} -> {out_path}")
            any_found = True
        except Exception as e:
            print(f"Failed {name}: {e}")

    if not any_found:
        print("No CSV files found in inputs/. Nothing to do.")


if __name__ == "__main__":
    main()
