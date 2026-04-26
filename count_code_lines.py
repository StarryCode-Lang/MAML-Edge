from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_EXTENSIONS = {".py", ".js", ".css", ".html", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}


def is_code_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in CODE_EXTENSIONS


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def main() -> None:
    file_counts: list[tuple[Path, int]] = []

    for path in ROOT.rglob("*"):
        if should_skip(path) or not is_code_file(path):
            continue

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            line_count = sum(1 for _ in f)

        file_counts.append((path.relative_to(ROOT), line_count))

    file_counts.sort(key=lambda item: str(item[0]))

    total_lines = 0
    print("Code line counts by file:")
    for rel_path, line_count in file_counts:
        total_lines += line_count
        print(f"{line_count:6}  {rel_path}")

    print(f"\nCode files: {len(file_counts)}")
    print(f"Total code lines: {total_lines}")


if __name__ == "__main__":
    main()
