#!/usr/bin/env python3
"""Create deterministic SHA-256 manifests for raw evidence and artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(destination: Path, paths: list[Path]) -> None:
    lines = [f"{digest(path)}  {path.relative_to(ROOT)}" for path in sorted(set(paths))]
    destination.write_text("\n".join(lines) + "\n")


def main() -> None:
    raw_paths = [p for base in (ROOT / "raw", ROOT / "logs") for p in base.rglob("*") if p.is_file()]
    write_manifest(ROOT / "RAW_MANIFEST.sha256", raw_paths)

    bases = ["analysis", "configs", "datasets", "figures", "paper", "processed", "prompts", "scripts", "tables", "vendor"]
    artifact_paths = []
    for name in bases:
        base = ROOT / name
        if base.exists():
            artifact_paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    artifact_paths.extend(
        ROOT / name for name in (
            "README.md", "ENVIRONMENT.md", "ETHICS_AND_DISCLOSURE.md",
            "PREREGISTRATION.md", "FREEZE.sha256", "RESEARCH_REPORT.md",
            "EXECUTIVE_SUMMARY.md", "REPRODUCE.md", "PAPER_OUTLINE.md",
            "CLAIMS_MATRIX.md", "LITERATURE_SEARCH.md", "DATA_DICTIONARY.md",
            "REVIEW_RESPONSE_2026-08-09.md",
            "REVIEW_RESPONSE_ROUND2_2026-08-09.md",
            "REVIEW_RESPONSE_ROUND3_2026-08-09.md",
            "UPSTREAM_ISSUE_25618_COMMENT_DRAFT.md",
            "RAW_MANIFEST.sha256",
        )
    )
    write_manifest(ROOT / "ARTIFACT_MANIFEST.sha256", [p for p in artifact_paths if p.is_file()])
    print(f"raw files: {len(raw_paths)}")
    print(f"artifact files: {len(artifact_paths)}")


if __name__ == "__main__":
    main()
