"""Pure Python module naming, shared by extraction and import resolution."""

from pathlib import PurePosixPath


def python_module_name(path: str, roots=("src", "lib")) -> str | None:
    pure = PurePosixPath(path)
    if pure.suffix not in {".py", ".pyi"}:
        return None
    stem = pure.with_suffix("").as_posix()
    for root in sorted((r.strip("/") for r in roots if r), key=len, reverse=True):
        if stem.startswith(root + "/"):
            stem = stem[len(root) + 1:]
            break
    parts = stem.split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None
