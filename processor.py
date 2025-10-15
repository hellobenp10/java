import os
import csv
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import re
from collections import defaultdict

BLOCK_WIDTH = 6

SEMANTIC_LABELS = {
    "Max","Mor","Mini","Grand","Super","Major",
    "g","h","i","d","e","f","a","b","c",
    "x2","x3","x4","x5","x6","x7","x8","x9","x10",
    "FG","ITS ALIVE","Notes"
}

FAMILY_CANONICAL_KEYS = {
    "credit": ("1CT", 0.60),
    "$1": ("$1", 10.0),
    "$2": ("$2", 20.0),
}

MULTS = [1,2,3,4,5,6,7,8,9,10]

Number = Optional[float]


def clean_number(val: Any) -> Optional[float]:
    try:
        s = str(val).strip().replace(",", "")
        if not s:
            return None
        if s.startswith("$"):
            s = s[1:]
        return float(s)
    except Exception:
        return None


def to_df(table: List[List[Any]]) -> pd.DataFrame:
    # Use first row as header if it looks like Row/Col header. Otherwise synthesize headers
    if not table:
        return pd.DataFrame()
    width = max(len(r) for r in table)
    norm = [r + [""]*(width-len(r)) for r in table]
    # build simple 0..N-1 headers
    headers = [f"C{i}" for i in range(width)]
    return pd.DataFrame(norm, columns=headers)


def slice_blocks(df: pd.DataFrame, block_width: int = BLOCK_WIDTH) -> List[Tuple[int,int]]:
    cols = list(range(df.shape[1]))
    return [(i, min(i+block_width-1, df.shape[1]-1)) for i in range(0, df.shape[1], block_width)]


def parse_block_denom(df: pd.DataFrame, start: int, end: int) -> Tuple[str, float, Tuple[int,int]]:
    """Find denomination token (1CT, $1, $2) and its numeric value within block; return key and coordinates."""
    for r in range(df.shape[0]):
        for c in range(start, end+1):
            cell = str(df.iat[r,c]).strip()
            if cell in {"1CT", "$1", "$2"}:
                nxt = str(df.iat[r, c+1]).strip() if c+1 <= end else ""
                val = clean_number(nxt) or 0.0
                return f"{cell} {nxt}".strip(), val, (r,c)
    return "", 0.0, (-1,-1)


def family_of(denom_key: str) -> str:
    if denom_key.startswith("1CT") or "CT" in denom_key:
        return "credit"
    if denom_key.startswith("$1"):
        return "$1"
    if denom_key.startswith("$2"):
        return "$2"
    return "unknown"


def detect_semantics(df: pd.DataFrame, start: int, end: int) -> Dict[Tuple[int,int], str]:
    sem: Dict[Tuple[int,int], str] = {}
    for r in range(df.shape[0]):
        for c in range(start, end+1):
            val = str(df.iat[r,c]).strip()
            if val in SEMANTIC_LABELS:
                sem[(r,c)] = val
    return sem


def extract_base_values_from_pair(df: pd.DataFrame,
                                  tmpl_start: int, tmpl_end: int,
                                  val_start: int, val_end: int,
                                  denom_value: float) -> Dict[str, Dict[str,float]]:
    """
    Extract base values where labels live in the template block (tmpl_*)
    and numbers live in the value block (val_*) at the same column offset.
    """
    base: Dict[str, Dict[str,float]] = {}
    keep = {"Max","Mor","Mini","g","h","i","d","e","f","a","b","c"}
    width = min(tmpl_end - tmpl_start, val_end - val_start)
    for r in range(df.shape[0]):
        for c in range(tmpl_start, tmpl_end+1):
            label = str(df.iat[r, c]).strip()
            if label in keep:
                offset = c - tmpl_start
                val_c = val_start + offset
                if val_start <= val_c <= val_end:
                    num = clean_number(df.iat[r, val_c])
                    if num is not None:
                        base[label] = {"credits": num, "cash": round(num*denom_value, 2)}
    return base


def find_canonical_per_family(df: pd.DataFrame, blocks: List[Tuple[int,int]]):
    """Choose canonical value blocks per family; semantics come from the left template block."""
    fam_to_canon = {}
    for idx, (start, end) in enumerate(blocks):
        key, denom, _ = parse_block_denom(df, start, end)
        fam = family_of(key)
        if fam == "unknown" or denom == 0:
            continue
        # template block is the immediate left block
        if idx == 0:
            continue
        tmpl_start, tmpl_end = blocks[idx-1]
        semantics = detect_semantics(df, tmpl_start, tmpl_end)
        bases = extract_base_values_from_pair(df, tmpl_start, tmpl_end, start, end, denom)
        has_core = all(k in bases for k in ["Max","Mor","Mini"]) if fam=="credit" else True
        canon_tok, _ = FAMILY_CANONICAL_KEYS.get(fam, (None, None))
        if canon_tok and key.startswith(canon_tok):
            fam_to_canon[fam] = (start, end, denom, key, semantics, bases, tmpl_start, tmpl_end)
            continue
        if fam not in fam_to_canon and bases:
            fam_to_canon[fam] = (start, end, denom, key, semantics, bases, tmpl_start, tmpl_end)
    return fam_to_canon


def scale_block_base(c_base: Dict[str,Dict[str,float]], c_denom: float, denom: float) -> Dict[str,Dict[str,float]]:
    if c_denom == 0:
        return c_base
    scale = denom / c_denom
    out = {}
    for sym,vals in c_base.items():
        credits = vals["credits"] * scale
        out[sym] = {
            "credits": round(credits,6),
            "cash": round(credits * denom, 6)
        }
    return out


# Base symbols that define numeric anchors
BASE_LABELS = {"Max","Mor","Mini","g","h","i","d","e","f","a","b","c"}


def parse_multiplier(label: str) -> Optional[int]:
    """Parse labels like 'x2', 'x10' to integer multiplier; else None."""
    m = re.fullmatch(r"x(\d+)", label.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def update_in_place(df: pd.DataFrame, mapping: Dict[str, Dict[str,Any]], blocks: List[Tuple[int,int]]) -> pd.DataFrame:
    updated = df.copy()
    for idx, (start, end) in enumerate(blocks):
        key, _, _ = parse_block_denom(df, start, end)
        info = mapping.get(key)
        if not info:
            continue
        base_vals = info.get("base_values", {})
        semantics = info.get("semantic_template", {})
        tmpl_start = info.get("template_start")
        tmpl_end = info.get("template_end")
        if tmpl_start is None:
            # fallback to immediate left block
            if idx > 0:
                tmpl_start, tmpl_end = blocks[idx-1]
            else:
                continue
        # write base numbers into value block columns aligned by offset from template
        for (r, c_label), label in semantics.items():
            if label in base_vals:
                offset = c_label - tmpl_start
                val_c = start + offset
                if start <= val_c <= end:
                    if info["family"] == "credit":
                        updated.iat[r, val_c] = base_vals[label]["credits"]
                    else:
                        num = base_vals[label]["credits"]
                        updated.iat[r, val_c] = f"${int(num) if float(num).is_integer() else num}"
        # compute derived xN values using nearest-left base label per row (in template), write into aligned value column
        row_positions: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for (r, c_label), lab in semantics.items():
            row_positions[r].append((c_label, lab))
        for r in row_positions:
            row_positions[r].sort(key=lambda t: t[0])
        for r, entries in row_positions.items():
            last_base_label: Optional[str] = None
            for c_label, lab in entries:
                if lab in BASE_LABELS and lab in base_vals:
                    last_base_label = lab
                    continue
                mult = parse_multiplier(lab)
                if mult and last_base_label:
                    base_num = base_vals.get(last_base_label, {}).get("credits")
                    if base_num is None:
                        continue
                    val = base_num * mult
                    offset = c_label - tmpl_start
                    val_c = start + offset
                    if start <= val_c <= end:
                        if info["family"] == "credit":
                            updated.iat[r, val_c] = round(val, 6)
                        else:
                            updated.iat[r, val_c] = f"${int(val) if float(val).is_integer() else val}"
    return updated


def process_table(table: List[List[Any]]) -> pd.DataFrame:
    df = to_df(table)
    blocks = slice_blocks(df)

    # Build per-block info and canonical references
    mapping: Dict[str, Dict[str,Any]] = {}
    fam_to_canon = find_canonical_per_family(df, blocks)

    for idx, (start,end) in enumerate(blocks):
        key, denom, _ = parse_block_denom(df, start, end)
        fam = family_of(key)
        # semantics come from template block on the left
        tmpl_start, tmpl_end = blocks[idx-1] if idx > 0 else (None, None)
        semantics = detect_semantics(df, tmpl_start, tmpl_end) if tmpl_start is not None else {}
        if fam in fam_to_canon:
            c_start,c_end,c_denom,c_key,c_sem,c_base,c_tmpl_start,c_tmpl_end = fam_to_canon[fam]
            # Regenerate base values for this block by scaling canonical base
            base_vals = scale_block_base(c_base, c_denom, denom)
            # Use canonical semantics template to propagate into all blocks of the family
            mapping[key] = {
                "family": fam,
                "denom_value": denom,
                "canonical_base": c_key,
                "scale_factor": denom / c_denom if c_denom else None,
                "semantic_template": c_sem,
                "template_start": tmpl_start,
                "template_end": tmpl_end,
                "base_values": base_vals,
            }
        else:
            # Unknown family or missing denom
            # try extracting from pair with left template if available
            if tmpl_start is not None:
                base_vals = extract_base_values_from_pair(df, tmpl_start, tmpl_end, start, end, denom)
            else:
                base_vals = {}
            mapping[key] = {
                "family": fam,
                "denom_value": denom,
                "canonical_base": None,
                "scale_factor": None,
                "semantic_template": semantics,
                "template_start": tmpl_start,
                "template_end": tmpl_end,
                "base_values": base_vals,
            }

    updated = update_in_place(df, mapping, blocks)
    return updated


def load_csv(path: str) -> List[List[str]]:
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def save_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, header=False)


def process_inputs(inputs_dir: str = "inputs", output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(inputs_dir):
        if not name.lower().endswith('.csv'):
            continue
        in_path = os.path.join(inputs_dir, name)
        table = load_csv(in_path)
        if not table:
            continue
        out_df = process_table(table)
        out_path = os.path.join(output_dir, f"processed_{name}")
        save_csv(out_path, out_df)
        print(f"Processed {name} -> {out_path}")


if __name__ == "__main__":
    if not os.path.exists("inputs"):
        os.makedirs("inputs", exist_ok=True)
        print("Created inputs/. Place CSVs inside and rerun.")
    process_inputs()
