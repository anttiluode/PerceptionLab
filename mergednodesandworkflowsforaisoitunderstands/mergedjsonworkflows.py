# mergeallfiles.py
import os
import argparse

def iter_py_files(folder, recursive=False):
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for name in filenames:
                if name.lower().endswith(".json"):
                    yield os.path.join(dirpath, name)
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(".json"):
                yield path

def read_text_best_effort(path):
    # Try common encodings; fall back to replacement to avoid crashing
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser(description="Concatenate .py files into one text file.")
    ap.add_argument("folder", help="Folder containing Python files (e.g. nodes)")
    ap.add_argument("output", help="Output text file path (e.g. combined.txt)")
    ap.add_argument("--recursive", action="store_true", help="Include subfolders")
    args = ap.parse_args()

    files = sorted(iter_py_files(args.folder, args.recursive))
    if not files:
        print("No .py files found.")
        return

    # Prevent accidental self-inclusion if output is inside the folder
    out_abs = os.path.abspath(args.output)

    with open(args.output, "w", encoding="utf-8") as out:
        for fp in files:
            if os.path.abspath(fp) == out_abs:
                continue
            rel = os.path.relpath(fp, args.folder)
            out.write(f"\n\n=== FILE: {rel} ===\n\n")
            out.write(read_text_best_effort(fp))

    print(f"Wrote {len(files)} Python files into: {args.output}")

if __name__ == "__main__":
    main()
