#!/usr/bin/env python3
"""Anonymization scanner for the CAFE repository.

Walks the publication-facing surface of the repo (src/, paper/, bench/,
README.md, notebooks/, pyproject.toml) and reports ``file:line`` hits for
de-anonymizing content: author name, email, the "Sovai" brand/affiliation,
absolute ``/Users/dereksnow`` paths, GitHub org URLs, and git-author metadata.

This is the exact leak class that embarrassed TSI-Bench: an author name plus a
hardcoded local path baked into a "blind" submission artifact. The intent here
is to catch those before they ship.

Report only -- this tool NEVER edits or redacts anything. It exits non-zero
when CRITICAL or HIGH findings exist so it can gate a release/CI step.

Usage:
    python tools/anonymization_scan.py            # scan default repo root
    python tools/anonymization_scan.py --root .   # explicit root
    python tools/anonymization_scan.py --json      # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Scan surface
# --------------------------------------------------------------------------- #
# Publication-facing paths to walk, relative to the repo root. Directories are
# recursed; individual files are scanned directly.
SCAN_TARGETS = [
    "src",
    "paper",
    "bench",
    "notebooks",
    "README.md",
    "pyproject.toml",
]

# Extensions worth scanning as text. .ipynb is handled specially (JSON cells).
TEXT_EXTS = {
    ".py", ".tex", ".md", ".toml", ".txt", ".cfg", ".ini",
    ".yaml", ".yml", ".rst", ".sh", ".bib", ".json",
}

# Never descend into these directory names.
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".ipynb_checkpoints", "node_modules", ".venv", "venv", ".vscode",
    ".benchmarks", "data",
}

# Binary artifacts we still want to inspect for embedded identity strings
# (e.g. a compiled PDF that may carry author metadata). Scanned byte-wise.
BINARY_SCAN_EXTS = {".pdf"}


# --------------------------------------------------------------------------- #
# Severity-ranked rules
# --------------------------------------------------------------------------- #
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    pattern: re.Pattern
    why: str


def _rules() -> list[Rule]:
    """De-anonymization detection rules, most damaging first.

    Patterns are intentionally specific to this project's known identity to
    keep false positives low; the literal-string forms double as a portable
    checklist if the repo is later renamed/anonymized.
    """
    return [
        # ---- CRITICAL: explicit publication-facing identity -------------- #
        Rule(
            "author-name", "CRITICAL",
            re.compile(r"Derek\s+Snow|\bDerek\b|\bSnow,\s*Derek\b"),
            "Real author name; de-anonymizes a blind submission / ships on PyPI.",
        ),
        Rule(
            "author-email", "CRITICAL",
            re.compile(
                r"cloudmachineengine@gmail\.com"
                r"|[A-Za-z0-9._%+-]+@(?:outlook|gmail|sovai)\.[A-Za-z]{2,}"
            ),
            "Personal/affiliation email address ties the artifact to the author.",
        ),
        Rule(
            "affiliation", "CRITICAL",
            re.compile(r"Sovai\s+Research|\\affil\b"),
            "Named affiliation in publication-facing source.",
        ),
        Rule(
            "latex-author", "CRITICAL",
            re.compile(r"\\author\b"),
            "LaTeX \\author macro -- compiles author identity into the PDF.",
        ),
        Rule(
            "pkg-author", "CRITICAL",
            re.compile(r"^\s*authors?\s*=", re.IGNORECASE),
            "Package metadata author field -- published with the package.",
        ),

        # ---- HIGH: org + username embedded in paths / URLs -------------- #
        Rule(
            "abs-user-path", "HIGH",
            re.compile(r"/Users/dereksnow"),
            "Absolute local path leaks the OS username; travels with the repo.",
        ),
        Rule(
            "org-brand", "HIGH",
            re.compile(r"\bSovai\b|\bsovai-research\b|\bsovai\b"),
            "Org/brand name attributes the artifact to a specific group.",
        ),
        Rule(
            "github-org-url", "HIGH",
            re.compile(r"github\.com/(?:sovai-research|dereksnow)\b"),
            "GitHub org/owner URL ties the package to the author identity.",
        ),

        # ---- MEDIUM: portability/structural identity leaks -------------- #
        Rule(
            "repo-abs-path", "MEDIUM",
            re.compile(r"/Users/[A-Za-z0-9._-]+/"),
            "Any absolute /Users/<name> path leaks a username and is non-portable.",
        ),

        # ---- LOW: incidental brand mentions ---------------------------- #
        Rule(
            "brand-comment", "LOW",
            re.compile(r"sovai-style|sovai_", re.IGNORECASE),
            "Incidental brand mention; contributes to org attribution.",
        ),
    ]


# --------------------------------------------------------------------------- #
# Finding model
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    severity: str
    rule: str
    path: str        # repo-relative
    line: int        # 1-based; 0 == file-level (e.g. binary/metadata)
    text: str
    why: str

    def sort_key(self) -> tuple:
        return (SEVERITY_ORDER.get(self.severity, 99), self.path, self.line)


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def _iter_files(root: Path) -> list[Path]:
    """Resolve SCAN_TARGETS into a concrete file list under root."""
    files: list[Path] = []
    for target in SCAN_TARGETS:
        p = root / target
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                fp = Path(dirpath) / fn
                ext = fp.suffix.lower()
                if ext in TEXT_EXTS or ext in BINARY_SCAN_EXTS:
                    files.append(fp)
    return files


def _scan_text_lines(rel: str, lines: list[str], rules: list[Rule],
                     result: ScanResult, self_path: str) -> None:
    """Apply every rule to each line and record findings."""
    # Never flag this scanner's own rule definitions as hits.
    is_self = rel == self_path
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        for rule in rules:
            if rule.pattern.search(line):
                if is_self:
                    continue
                snippet = line.strip()
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                result.findings.append(Finding(
                    severity=rule.severity, rule=rule.name, path=rel,
                    line=i, text=snippet, why=rule.why,
                ))


def _scan_notebook(rel: str, raw: str, rules: list[Rule],
                   result: ScanResult, self_path: str) -> None:
    """Scan .ipynb: cell source AND notebook metadata (author fields)."""
    try:
        nb = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Fall back to treating it as text so we still catch literals.
        _scan_text_lines(rel, raw.splitlines(), rules, result, self_path)
        return

    # Metadata block (kernelspec, authors, etc.).
    meta = json.dumps(nb.get("metadata", {}))
    for rule in rules:
        if rule.pattern.search(meta):
            result.findings.append(Finding(
                severity=rule.severity, rule=rule.name, path=rel,
                line=0, text=f"[notebook metadata] {meta[:160]}", why=rule.why,
            ))

    # Cell sources, with best-effort line attribution.
    for ci, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(keepends=True)
        for li, raw_line in enumerate(src, start=1):
            line = raw_line.rstrip("\n")
            for rule in rules:
                if rule.pattern.search(line):
                    result.findings.append(Finding(
                        severity=rule.severity, rule=rule.name, path=rel,
                        line=0, text=f"[cell {ci} line {li}] {line.strip()[:160]}",
                        why=rule.why,
                    ))


def _scan_binary(rel: str, data: bytes, rules: list[Rule],
                 result: ScanResult) -> None:
    """Byte-scan a binary artifact (e.g. PDF) for embedded identity strings.

    Only literal-text rules make sense here; we reuse the same patterns against
    a latin-1 decode (lossless byte->char) and report at file level.
    """
    text = data.decode("latin-1", errors="ignore")
    seen: set[str] = set()
    for rule in rules:
        m = rule.pattern.search(text)
        if m and rule.name not in seen:
            seen.add(rule.name)
            frag = m.group(0)[:80]
            result.findings.append(Finding(
                severity=rule.severity, rule=rule.name, path=rel,
                line=0, text=f"[binary embedded] match={frag!r}", why=rule.why,
            ))


def scan_repo(root: Path) -> ScanResult:
    rules = _rules()
    result = ScanResult()
    self_path = "tools/anonymization_scan.py"
    for fp in _iter_files(root):
        rel = os.path.relpath(fp, root)
        ext = fp.suffix.lower()
        try:
            if ext in BINARY_SCAN_EXTS:
                _scan_binary(rel, fp.read_bytes(), rules, result)
            else:
                raw = fp.read_text(encoding="utf-8", errors="replace")
                if ext == ".ipynb":
                    _scan_notebook(rel, raw, rules, result, self_path)
                else:
                    _scan_text_lines(rel, raw.splitlines(), rules, result,
                                     self_path)
            result.files_scanned += 1
        except OSError as exc:  # pragma: no cover - defensive
            result.errors.append(f"{rel}: {exc}")
    return result


# --------------------------------------------------------------------------- #
# Git-author metadata check (separate axis: not a file, but ships in history)
# --------------------------------------------------------------------------- #
def scan_git_identity(root: Path) -> list[Finding]:
    """Report git author identity, which travels in every committed object.

    A blind artifact built from this repo still carries the committer name/email
    in .git history -- the second half of the TSI-Bench leak.
    """
    findings: list[Finding] = []
    rules = _rules()
    name_re = next(r.pattern for r in rules if r.name == "author-name")
    email_re = next(r.pattern for r in rules if r.name == "author-email")
    for key, label, sev in [
        ("user.name", "git config user.name", "HIGH"),
        ("user.email", "git config user.email", "HIGH"),
    ]:
        try:
            out = subprocess.run(
                ["git", "config", "--get", key],
                cwd=root, capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        val = out.stdout.strip()
        if not val:
            continue
        if name_re.search(val) or email_re.search(val) \
                or "@" in val:
            findings.append(Finding(
                severity=sev, rule="git-author-metadata",
                path="<git>", line=0, text=f"{label} = {val}",
                why="Author identity recorded in git history; ships with the repo.",
            ))
    return findings


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(result: ScanResult, git_findings: list[Finding]) -> None:
    all_findings = sorted(
        result.findings + git_findings, key=lambda f: f.sort_key()
    )
    counts: dict[str, int] = {}
    for f in all_findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    print("=" * 78)
    print("ANONYMIZATION SCAN  --  CAFE repository (report only, no redaction)")
    print("=" * 78)
    print(f"files scanned: {result.files_scanned}    "
          f"total findings: {len(all_findings)}")
    summary = "  ".join(
        f"{sev}={counts.get(sev, 0)}"
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    )
    print(f"by severity:   {summary}")
    if result.errors:
        print(f"read errors:   {len(result.errors)}")
    print()

    if not all_findings:
        print("No de-anonymizing content found in the scanned surface.")
        return

    current_sev = None
    for f in all_findings:
        if f.severity != current_sev:
            current_sev = f.severity
            print("-" * 78)
            print(f"[{current_sev}]")
            print("-" * 78)
        loc = f"{f.path}:{f.line}" if f.line else f.path
        print(f"  {loc}")
        print(f"      rule : {f.rule}")
        print(f"      hit  : {f.text}")
        print(f"      why  : {f.why}")
    print("-" * 78)
    print("Remediation is intentionally NOT automated. Review each hit; replace "
          "absolute paths\nwith repo-relative / os.path-derived paths, and strip "
          "author/affiliation/email\nfrom publication-facing files before release.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=None,
        help="Repo root to scan (default: parent of tools/).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the text report.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root \
        else Path(__file__).resolve().parent.parent

    result = scan_repo(root)
    git_findings = scan_git_identity(root)
    all_findings = sorted(
        result.findings + git_findings, key=lambda f: f.sort_key()
    )

    if args.json:
        print(json.dumps({
            "files_scanned": result.files_scanned,
            "errors": result.errors,
            "findings": [
                {
                    "severity": f.severity, "rule": f.rule, "path": f.path,
                    "line": f.line, "text": f.text, "why": f.why,
                }
                for f in all_findings
            ],
        }, indent=2))
    else:
        print_report(result, git_findings)

    # Exit non-zero on CRITICAL/HIGH so this can gate CI / a release step.
    blocking = any(f.severity in ("CRITICAL", "HIGH") for f in all_findings)
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
