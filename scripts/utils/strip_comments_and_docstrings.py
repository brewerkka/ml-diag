
from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

SKIP_DIR_NAMES = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "node_modules",
}

                                                                    
                    
DIRECTIVE_PATTERNS = re.compile(
    r"#\s*(type:\s*ignore|noqa|pragma:|fmt:\s*(?:off|on)|isort:|"
    r"pyright:|mypy:|ruff:|nosec|coverage:)",
    re.IGNORECASE,
)


def _is_directive(comment: str) -> bool:
    return bool(DIRECTIVE_PATTERNS.search(comment))


def _find_docstring_line_ranges(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()

    def _maybe_docstring(node) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
                                                                          
            start = first.lineno
            end = getattr(first, "end_lineno", start) or start
            for ln in range(start, end + 1):
                lines.add(ln)

    _maybe_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _maybe_docstring(node)
    return lines


def _strip_comments(source: str) -> str:
    result: list[str] = []
    last_end = (1, 0)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return source
    for tok in tokens:
        tok_type, tok_str, start, end, _ = tok
        if tok_type == tokenize.COMMENT:
            if _is_directive(tok_str):
                                                      
                                                         
                if start[0] == last_end[0]:
                    gap = start[1] - last_end[1]
                    if gap > 0:
                        result.append(" " * gap)
                else:
                    result.append("\n" * (start[0] - last_end[0]))
                    if start[1] > 0:
                        result.append(" " * start[1])
                result.append(tok_str)
                last_end = end
            else:
                                                                    
                                                              
                last_end = end
            continue
                                                                        
                                          
        if start[0] == last_end[0]:
            gap = start[1] - last_end[1]
            if gap > 0:
                result.append(" " * gap)
        else:
            result.append("\n" * (start[0] - last_end[0]))
            if start[1] > 0:
                result.append(" " * start[1])
        if tok_str:
            result.append(tok_str)
        last_end = end
    return "".join(result)


def _strip_docstrings(source: str) -> str:
    doc_lines = _find_docstring_line_ranges(source)
    if not doc_lines:
        return source
    lines = source.splitlines(keepends=True)
                                                                           
                                                              
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    pass_lines: dict[int, str] = {}                           

    def _check_body(node) -> None:
        body = getattr(node, "body", None)
        if not body or len(body) != 1:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
                                                                 
                                                             
            indent = " " * first.col_offset
                                                                  
            target_line = (getattr(first, "end_lineno", first.lineno) or first.lineno) - 1
            pass_lines[target_line] = f"{indent}pass\n"

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _check_body(node)
                                                                            
    out: list[str] = []
    for i, line in enumerate(lines):
        line_no = i + 1           
        if line_no in doc_lines:
            if i in pass_lines:
                out.append(pass_lines[i])
            continue
        out.append(line)
    return "".join(out)


def _strip_file(text: str) -> str:
    text = _strip_comments(text)
    text = _strip_docstrings(text)
                                                   
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _should_process(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    parts = set(path.parts)
    if parts & SKIP_DIR_NAMES:
        return False
                                                                    
    if "_archive" in parts:
        return False
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--apply", action="store_true", help="Overwrite files in place (default: dry-run)"
    )
    p.add_argument(
        "--root",
        default=str(ROOT),
        type=Path,
        help="Root directory to scan (default: project root)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root: Path = args.root
    files = sorted(p for p in root.rglob("*.py") if _should_process(p))
    print(f"Scanning {len(files)} Python files under {root}")
    print(f"Mode: {'APPLY (overwriting)' if args.apply else 'DRY-RUN'}")
    print()
    total_chars_before = 0
    total_chars_after = 0
    total_changed = 0
    failed: list[tuple[Path, str]] = []
    for path in files:
        try:
            original = path.read_text(encoding="utf-8")
        except Exception as e:
            failed.append((path, str(e)))
            continue
        try:
            stripped = _strip_file(original)
        except Exception as e:
            failed.append((path, f"strip failed: {type(e).__name__}: {e}"))
            continue
                                                        
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            failed.append((path, f"syntax broken after strip: {e}"))
            continue
        total_chars_before += len(original)
        total_chars_after += len(stripped)
        if stripped != original:
            total_changed += 1
            if args.apply:
                path.write_text(stripped, encoding="utf-8")
    print(f"Files scanned:  {len(files)}")
    print(f"Files changed:  {total_changed}")
    print(f"Files failed:   {len(failed)}")
    print(f"Size before:    {total_chars_before:,} chars")
    print(f"Size after:     {total_chars_after:,} chars")
    delta = total_chars_before - total_chars_after
    print(f"Reduction:      {delta:,} chars ({100 * delta / max(total_chars_before, 1):.1f}%)")
    if failed:
        print()
        print("FAILED files (left untouched):")
        for path, err in failed:
            print(f"  - {path.relative_to(root)}: {err}")
    if not args.apply:
        print()
        print("DRY-RUN complete. Re-run with --apply to overwrite files.")
        print("RECOMMENDED: commit your current state to git first.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
