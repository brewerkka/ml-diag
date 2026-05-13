from __future__ import annotations

import argparse
import io
import sys
import tokenize
from pathlib import Path

SKIP_DIRS = {
    ".venv",
    "__pycache__",
    "_archive",
    ".git",
    "build",
    "dist",
    ".ruff_cache",
    ".pytest_cache",
}


def _protected_lines(src: str) -> set[int]:
    protected: set[int] = set()
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    try:
        toks = tokenize.tokenize(io.BytesIO(src.encode("utf-8")).readline)
        for tok in toks:
            if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
                for ln in range(tok.start[0], tok.end[0] + 1):
                    protected.add(ln)
            elif (
                fstring_middle is not None
                and tok.type == fstring_middle
                and tok.start[0] != tok.end[0]
            ):
                for ln in range(tok.start[0], tok.end[0] + 1):
                    protected.add(ln)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    return protected


def normalize(src: str) -> str:
    lines = src.split("\n")
    protected = _protected_lines(src)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line_no = i + 1
        if line_no in protected:
            out.append(lines[i])
            i += 1
            continue
        line = lines[i]
        if line.strip() == "":
            j = i
            while j < n and lines[j].strip() == "" and (j + 1) not in protected:
                j += 1
            if j >= n:
                i = j
                continue
            nxt = lines[j]
            if nxt.startswith((" ", "\t")):
                i = j
                continue
            else:
                cap = min(j - i, 2)
                if not out:
                    cap = 0
                out.extend([""] * cap)
                i = j
                continue
        else:
            out.append(line.rstrip())
            i += 1
    while out and out[-1] == "":
        out.pop()
    out.append("")
    return "\n".join(out)


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    changed = 0
    failed: list[tuple[Path, str]] = []
    total = 0
    for root in args.roots:
        for path in _iter_py_files(root):
            total += 1
            try:
                src = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                failed.append((path, f"read: {e}"))
                continue
            new = normalize(src)
            if new == src:
                continue
            changed += 1
            if args.apply:
                try:
                    path.write_text(new, encoding="utf-8")
                except OSError as e:
                    failed.append((path, f"write: {e}"))
    mode = "WRITE" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanned={total} changed={changed} failed={len(failed)}")
    for p, why in failed:
        print(f"  FAIL {p}: {why}", file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
