"""Path / config / table IO. All paths are relative to the repository root."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover - sandbox fallback, no network install
    yaml = None  # type: ignore
    _HAS_YAML = False

# Repository root is derived from this file's location, never hard-coded.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Minimal YAML-subset reader used ONLY when PyYAML is unavailable. It supports
# exactly the constructs used by this repo's configs: nested mappings, lists,
# inline maps {k: v}, inline lists [a, b], quoted strings and numeric literals.
# In a normal environment (requirements.txt installed) PyYAML is used instead.
# ---------------------------------------------------------------------------
def _strip_comment(line: str) -> str:
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _parse_scalar(text: str):
    s = text.strip()
    if not s:
        return None
    if s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    if s.startswith("{") and s.endswith("}"):
        return _parse_inline_map(s[1:-1])
    if s.startswith("[") and s.endswith("]"):
        return _parse_inline_list(s[1:-1])
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts, depth, buf, quote = [], 0, [], None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "{[":
            depth += 1
            buf.append(ch)
        elif ch in "}]":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_inline_map(inner: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for token in _split_top(inner):
        if not token.strip():
            continue
        k, sep, v = token.partition(":")
        if sep:
            out[k.strip()] = _parse_scalar(v)
    return out


def _parse_inline_list(inner: str) -> list:
    return [_parse_scalar(t) for t in _split_top(inner) if t.strip()]


def _mini_yaml_load(text: str) -> Dict[str, Any]:
    raw: list[tuple[int, str]] = []
    for line in text.splitlines():
        clean = _strip_comment(line).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        raw.append((indent, clean.strip()))

    def parse_block(i: int, indent: int):
        is_list = raw[i][1].startswith("- ")
        container: Any = [] if is_list else {}
        while i < len(raw) and raw[i][0] == indent:
            content = raw[i][1]
            if is_list:
                if not content.startswith("- "):
                    break
                container.append(_parse_scalar(content[2:]))
                i += 1
            else:
                if content.startswith("- "):
                    break
                k, sep, v = content.partition(":")
                key = k.strip()
                if not sep:
                    raise ValueError(f"Malformed YAML line: {content!r}")
                if v.strip():
                    container[key] = _parse_scalar(v)
                    i += 1
                else:
                    if i + 1 < len(raw) and raw[i + 1][0] > indent:
                        child, i = parse_block(i + 1, raw[i + 1][0])
                        container[key] = child
                    else:
                        container[key] = None
                        i += 1
        return container, i

    result, _ = parse_block(0, raw[0][0])
    return result


def resolve_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a user-supplied path; relative paths are rooted at REPO_ROOT.

    Absolute paths are rejected on purpose: the artifact must run from any
    checkout location and must not depend on a particular machine.
    """
    p = Path(path)
    if p.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed in this artifact: {path!r}. "
            "Pass a path relative to the repository root."
        )
    full = (REPO_ROOT / p).resolve()
    if must_exist and not full.exists():
        raise FileNotFoundError(f"Required path does not exist: {full}")
    return full


def load_yaml(rel_path: str | Path) -> Dict[str, Any]:
    path = resolve_path(rel_path, must_exist=True)
    with path.open("r", encoding="utf-8") as fh:
        text = fh.read()
    if _HAS_YAML:
        data = yaml.safe_load(text)
    else:
        data = _mini_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {rel_path}")
    return data


def ensure_dir(rel_path: str | Path) -> Path:
    path = resolve_path(rel_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(rel_path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> Path:
    """Write a small CSV (metrics/tables only; never replicate-level dumps here)."""
    path = resolve_path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = fieldnames or []
    else:
        fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_csv(rel_path: str | Path) -> List[Dict[str, str]]:
    path = resolve_path(rel_path, must_exist=True)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def find_repo_root(start: str | Path | None = None) -> Path:
    """Locate the repository root by walking up for a configs/ directory."""
    cur = Path(start).resolve() if start else Path(__file__).resolve().parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs").is_dir():
            return candidate
    return REPO_ROOT


def env_summary(packages: tuple[str, ...] = ("numpy", "pandas", "matplotlib", "PyYAML", "torch")) -> Dict[str, str]:
    """Collect installed versions for the run log (no absolute paths emitted)."""
    import importlib

    module_names = {"PyYAML": "yaml"}
    out: Dict[str, str] = {}
    for name in packages:
        try:
            mod = importlib.import_module(module_names.get(name, name))
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = "not-installed"
    out["python"] = os.sys.version.split()[0]
    return out
