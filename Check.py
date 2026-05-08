"""
Read and export a BPB checkpoint (.npz) produced by ELVTF optimizers.

Usage (from the same Python/interpreter where numpy/pandas are installed):
  python read_bpb_checkpoint.py "N:\\path\\to\\bpb_checkpoint.npz"
  python read_bpb_checkpoint.py "bpb_checkpoint.npz" -o "bpb_contents.xlsx"
  python read_bpb_checkpoint.py "bpb_checkpoint.npz" --extract gbest best_per_iter

Notes:
- The script uses np.load(..., allow_pickle=True) because many checkpoints contain object arrays.
  Only open .npz files you trust.
- If you get ImportError for numpy, run this script with the same python executable used by the optimizer
  (see sys.executable printed below) and ensure numpy/pandas are installed.

Outputs:
- Prints a short summary to stdout (keys, types, shapes, sample).
- Optional Excel export with one sheet per key (--out).
- Optional extraction of specific keys to CSV (--extract).
"""

import sys
import os
import argparse
import json
from typing import Any

try:
    import numpy as np
    import pandas as pd
except Exception as e:
    # Helpful message if numpy/pandas cannot be imported
    print("ERROR importing numpy/pandas:", e, file=sys.stderr)
    print("Make sure you run this with the same Python interpreter used by your optimizer.", file=sys.stderr)
    print("Python executable:", sys.executable, file=sys.stderr)
    raise

def safe_to_dataframe(key: str, obj: Any, max_rows: int | None = 2000):
    """
    Convert array/object to pandas DataFrame when reasonable.
    Returns (df, note) or (None, note) if conversion not possible.
    """
    note = ""
    # numpy numeric array
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # try to convert list-like elements
            try:
                lst = obj.tolist()
            except Exception:
                return None, "object array not convertible to list"
            if all(isinstance(x, dict) for x in lst):
                return pd.DataFrame(lst), "object array -> DataFrame(list-of-dict)"
            if all(not isinstance(x, (list, tuple, np.ndarray, dict)) for x in lst):
                # scalar list
                df = pd.DataFrame({key: lst})
                if max_rows and len(df) > max_rows:
                    return df.iloc[:max_rows].copy(), f"truncated to {max_rows}"
                return df, "object array scalar list -> DataFrame"
            # try to coerce to 2D numeric
            try:
                arr = np.array(lst)
                if arr.ndim == 1:
                    df = pd.DataFrame(arr, columns=[key])
                    if max_rows and len(df) > max_rows:
                        return df.iloc[:max_rows].copy(), f"truncated to {max_rows}"
                    return df, "object array coerced to 1D"
                if arr.ndim == 2:
                    cols = [f"{key}_{i}" for i in range(arr.shape[1])]
                    df = pd.DataFrame(arr, columns=cols)
                    if max_rows and len(df) > max_rows:
                        return df.iloc[:max_rows].copy(), f"truncated to {max_rows}"
                    return df, "object array coerced to 2D"
            except Exception:
                pass
            # fallback: stringify items
            ser = pd.Series([json.dumps(x, default=str, ensure_ascii=False) for x in lst], name=key)
            if max_rows and len(ser) > max_rows:
                return ser.iloc[:max_rows].to_frame(), f"object array -> strings (truncated to {max_rows})"
            return ser.to_frame(), "object array -> strings"
        else:
            # numeric dtype
            if obj.ndim == 0:
                return pd.DataFrame({key: [obj.item()]}), "0-d numeric"
            if obj.ndim == 1:
                df = pd.DataFrame(obj, columns=[key])
                if max_rows and len(df) > max_rows:
                    return df.iloc[:max_rows].copy(), f"truncated to {max_rows}"
                return df, "1-d numeric"
            if obj.ndim == 2:
                cols = [f"{key}_{i}" for i in range(obj.shape[1])]
                df = pd.DataFrame(obj, columns=cols)
                if max_rows and len(df) > max_rows:
                    return df.iloc[:max_rows].copy(), f"truncated to {max_rows}"
                return df, "2-d numeric"
            # higher dims -> flatten last dims
            flat = obj.reshape(obj.shape[0], -1)
            cols = [f"{key}_{i}" for i in range(flat.shape[1])]
            df = pd.DataFrame(flat, columns=cols)
            if max_rows and len(df) > max_rows:
                return df.iloc[:max_rows].copy(), f"flattened and truncated to {max_rows}"
            return df, f"flattened from {obj.ndim}-D numeric"
    # pandas objects
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return (obj if isinstance(obj, pd.DataFrame) else obj.to_frame()), "pandas object"
    # python list/dict
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return pd.DataFrame(obj), "list of dicts"
        try:
            return pd.DataFrame(obj), "list -> DataFrame"
        except Exception:
            ser = pd.Series([json.dumps(x, default=str, ensure_ascii=False) for x in obj], name=key)
            return ser.to_frame(), "list -> strings"
    if isinstance(obj, dict):
        try:
            return pd.DataFrame([obj]), "dict -> single-row DataFrame"
        except Exception:
            return None, "dict not convertible"
    # scalar fallback
    try:
        return pd.DataFrame({key: [obj]}), "scalar fallback"
    except Exception:
        return None, "unhandled type"

def inspect_npz(path: str, sample_items: int = 6):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print("Python executable:", sys.executable)
    print("Loading:", path)
    print("WARNING: This will use allow_pickle=True — only load trusted files.")
    with np.load(path, allow_pickle=True) as npz:
        keys = npz.files
        print("Contained keys:", keys)
        summary = {}
        for k in keys:
            v = npz[k]
            t = type(v)
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            print(f"\nKey: {k}\n  type={t}\n  shape={shape}\n  dtype={dtype}")
            sample = None
            try:
                if isinstance(v, np.ndarray):
                    if v.size == 0:
                        sample = "(empty array)"
                    else:
                        flat = v.flat
                        sample = [flat[i] for i in range(min(sample_items, v.size))]
                else:
                    sample = repr(v)[:1000]
            except Exception as e:
                sample = f"(sample failed: {e})"
            print("  sample:", sample)
            summary[k] = {"type": str(t), "shape": shape, "dtype": str(dtype), "sample": sample}
    return summary

def export_to_excel(npz_path: str, out_xlsx: str, max_rows: int | None = 2000):
    if not out_xlsx.lower().endswith((".xlsx", ".xls")):
        raise ValueError("Output must be an Excel file with extension .xlsx or .xls")
    print(f"Exporting '{npz_path}' -> '{out_xlsx}' (max_rows per sheet = {max_rows})")
    with np.load(npz_path, allow_pickle=True) as npz, pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for k in npz.files:
            v = npz[k]
            df, note = safe_to_dataframe(k, v, max_rows)
            if df is None:
                # fallback: write repr as single-cell table
                try:
                    repr_val = repr(v)
                    df = pd.DataFrame([{k: repr_val}])
                    note = "fallback: saved repr"
                except Exception:
                    df = pd.DataFrame([{k: str(type(v))}])
                    note = "fallback: type string"
            sheet = k[:31]
            # avoid existing sheet name collisions
            base = sheet
            i = 1
            while sheet in writer.sheets:
                sheet = f"{base[:28]}_{i}"
                i += 1
            df.to_excel(writer, sheet_name=sheet, index=False)
            meta = pd.DataFrame([{"key": k, "note": note, "shape": str(getattr(v, "shape", "")), "dtype": str(getattr(v, "dtype", ""))}])
            meta_sheet = f"{sheet}_meta"[:31]
            meta.to_excel(writer, sheet_name=meta_sheet, index=False)
            print(f"  wrote sheet '{sheet}' (note: {note})")

def extract_keys(npz_path: str, keys: list[str], out_dir: str):
    """Extract named keys to CSV files in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with np.load(npz_path, allow_pickle=True) as npz:
        for k in keys:
            if k not in npz.files:
                print(f"Key '{k}' not present in checkpoint.")
                continue
            v = npz[k]
            df, note = safe_to_dataframe(k, v, max_rows=None)
            if df is None:
                # save textual repr
                with open(os.path.join(out_dir, f"{k}.txt"), "w", encoding="utf-8") as f:
                    f.write(repr(v))
                print(f"  extracted '{k}' as text -> {k}.txt (note: {note})")
            else:
                out_csv = os.path.join(out_dir, f"{k}.csv")
                df.to_csv(out_csv, index=False)
                print(f"  extracted '{k}' -> {out_csv} (note: {note})")

def main():
    parser = argparse.ArgumentParser(description="Read and export a BPB checkpoint (.npz)")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint file (e.g. bpb_checkpoint.npz)")
    parser.add_argument("-o", "--out", help="Optional Excel file to write contents (e.g. bpb_contents.xlsx)")
    parser.add_argument("--maxrows", type=int, default=2000, help="Max rows per sheet when exporting (0 or negative = no limit)")
    parser.add_argument("--sample", type=int, default=6, help="Number of sample values printed per key")
    parser.add_argument("--extract", nargs="+", help="List of specific keys to extract to CSV files under ./extracted/")
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt):
        print("Checkpoint not found:", ckpt, file=sys.stderr)
        sys.exit(2)

    try:
        inspect_npz(ckpt, sample_items=args.sample)
    except Exception as e:
        print("Failed to inspect checkpoint:", e, file=sys.stderr)
        raise

    if args.out:
        max_rows = None if args.maxrows <= 0 else args.maxrows
        try:
            export_to_excel(ckpt, os.path.abspath(args.out), max_rows)
        except Exception as e:
            print("Export to Excel failed:", e, file=sys.stderr)
            raise

    if args.extract:
        outdir = os.path.join(os.path.dirname(ckpt), "extracted")
        try:
            extract_keys(ckpt, args.extract, outdir)
        except Exception as e:
            print("Extraction failed:", e, file=sys.stderr)
            raise
    print("Done.")

if __name__ == "__main__":
    main()