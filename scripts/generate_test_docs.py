#!/usr/bin/env python3
"""Generate docs/test-documentation.md from tests/test_*.py modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
OUT = ROOT / "docs" / "test-documentation.md"

DEFAULT_TARGET = "Not specified in test module"
DEFAULT_METHOD = "Uses unittest assertions around module behavior."

REWRITE = {
    "api": "API",
    "cdn": "CDN",
    "cli": "CLI",
    "doi": "DOI",
    "env": "environment",
    "http": "HTTP",
    "id": "ID",
    "ids": "IDs",
    "json": "JSON",
    "minio": "MinIO",
    "mcp": "MCP",
    "pdf": "PDF",
    "pdfs": "PDFs",
    "qdrant": "Qdrant",
    "redis": "Redis",
    "ui": "UI",
    "url": "URL",
    "urls": "URLs",
    "v2": "v2",
}


@dataclass(frozen=True)
class TestDoc:
    name: str
    lineno: int
    what: str
    how: str


@dataclass(frozen=True)
class ModuleDoc:
    path: Path
    target: str
    method: str
    tests: list[TestDoc]


def _string_const(module: ast.Module, name: str) -> str | None:
    for node in module.body:
        value = None
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                value = node.value
        if value is None:
            continue
        try:
            resolved = ast.literal_eval(value)
        except Exception:
            continue
        if isinstance(resolved, str):
            return resolved.strip()
    return None


def _what_from_name(name: str) -> str:
    name = name[5:] if name.startswith("test_") else name
    words = [REWRITE.get(token, token) for token in name.split("_")]
    phrase = " ".join(words).strip()
    return f"Verifies {phrase}." if phrase else "Verifies expected behavior."


def _techniques(source: str, decorators: list[str]) -> list[str]:
    parts: list[str] = []
    dec = " ".join(decorators).lower()
    low = source.lower()

    if "mock.patch" in source:
        parts.append("patches collaborators with unittest.mock")
    if "testclient(" in low:
        parts.append("exercises FastAPI routes with TestClient")
    if "httpx." in low:
        parts.append("issues live HTTP requests with httpx")
    if "temporarydirectory" in low:
        parts.append("builds temporary filesystem fixtures")
    if "redirect_stdout" in source or "redirect_stderr" in source:
        parts.append("captures CLI output streams")
    if "os.environ" in source:
        parts.append("overrides environment variables")
    if "sys.argv" in source:
        parts.append("simulates CLI argv input")
    if "skipif" in dec or "skipunless" in dec:
        parts.append("is gated by unittest skip decorators")

    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _how_text(base: str, source: str, decorators: list[str]) -> str:
    tech = _techniques(source, decorators)
    if not tech:
        return base
    return f"{base} Techniques in this test: {', '.join(tech)}."


def _collect_tests(module: ast.Module) -> list[tuple[ast.FunctionDef, list[str]]]:
    out: list[tuple[ast.FunctionDef, list[str]]] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            out.append((node, [ast.unparse(d) for d in node.decorator_list]))
        if isinstance(node, ast.ClassDef):
            class_decorators = [ast.unparse(d) for d in node.decorator_list]
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                    dec = class_decorators + [ast.unparse(d) for d in child.decorator_list]
                    out.append((child, dec))
    out.sort(key=lambda item: item[0].lineno)
    return out


def _module_doc(path: Path) -> ModuleDoc:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    target = _string_const(tree, "TEST_DOC_TARGET") or DEFAULT_TARGET
    method = _string_const(tree, "TEST_DOC_METHOD") or DEFAULT_METHOD
    tests: list[TestDoc] = []
    for fn, decorators in _collect_tests(tree):
        fn_src = ast.get_source_segment(src, fn) or ""
        tests.append(
            TestDoc(
                name=fn.name,
                lineno=fn.lineno,
                what=_what_from_name(fn.name),
                how=_how_text(method, fn_src, decorators),
            )
        )
    return ModuleDoc(path=path.relative_to(ROOT), target=target, method=method, tests=tests)


def _render(modules: list[ModuleDoc]) -> str:
    total = sum(len(m.tests) for m in modules)
    lines = [
        "# Test Documentation",
        "",
        "Generated from test modules with `python scripts/generate_test_docs.py`.",
        "",
        f"- Test files documented: {len(modules)}",
        f"- Individual tests documented: {total}",
        "",
    ]
    for m in modules:
        lines.extend(
            [
                f"## `{m.path.as_posix()}`",
                f"- Target: `{m.target}`",
                f"- Baseline method: {m.method}",
                f"- Tests: {len(m.tests)}",
                "",
            ]
        )
        for t in m.tests:
            lines.extend(
                [
                    f"### `{t.name}`",
                    f"- What: {t.what}",
                    f"- How: {t.how}",
                    f"- Location: `{m.path.as_posix()}:{t.lineno}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    modules = [_module_doc(p) for p in sorted(TESTS_DIR.glob("test_*.py"))]
    OUT.write_text(_render(modules), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
