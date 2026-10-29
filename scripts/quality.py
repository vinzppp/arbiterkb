"""E1 minimalism checks (DEVELOPMENT.md §1): module and function size, inheritance depth,
dependency whitelist. sg_compat.py is a verbatim copy and is exempt from the size limits."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_MODULE, MAX_FUNC = 250, 40
EXEMPT = {"sg_compat.py"}
ALLOWED_BASES = {"Protocol", "Exception", "RuntimeError", "ValueError", "object"}
WHITELIST = {"numpy", "scipy", "matplotlib", "pytest", "nbformat", "nbconvert", "arbiter_kb", "run", "selftest", "figures", "measure_kb", "dry_run", "video_windows"}  # last five: sibling scripts


def files() -> list[Path]:
    return sorted(p for d in ("arbiter_kb", "scripts", "tests") for p in (ROOT / d).rglob("*.py"))


def check_file(p: Path) -> list[str]:
    src = p.read_text()
    tree = ast.parse(src)
    errs, rel = [], p.relative_to(ROOT)
    exempt = p.name in EXEMPT or p.parts[-2] == "tests"
    if not exempt and len(src.splitlines()) > MAX_MODULE:
        errs.append(f"{rel}: {len(src.splitlines())} lines > {MAX_MODULE}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not exempt:
            n = node.end_lineno - node.lineno + 1
            if n > MAX_FUNC:
                errs.append(f"{rel}:{node.lineno} {node.name}() is {n} lines > {MAX_FUNC}")
        if isinstance(node, ast.ClassDef) and p.name not in EXEMPT:
            bad = [ast.unparse(b) for b in node.bases if ast.unparse(b) not in ALLOWED_BASES]
            if bad:
                errs.append(f"{rel}:{node.lineno} class {node.name} inherits {bad}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                top = m.split(".")[0]
                if top and getattr(node, "level", 0) == 0 and top not in sys.stdlib_module_names | WHITELIST:
                    errs.append(f"{rel}:{node.lineno} imports {top} (not in the dependency whitelist)")
    return errs


def main() -> int:
    errs = [e for p in files() for e in check_file(p)]
    for e in errs:
        print("[FAIL]", e)
    n = len(files())
    print(f"quality: {n} files, {len(errs)} problems" + (" — PASS" if not errs else " — FAIL"))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
