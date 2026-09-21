"""Rewrite qpocket_pipeline/qpocket_screen references to the qpocket layout."""
import re
import sys
from pathlib import Path

# Order matters: specific module paths before the bare package name.
MAP = [
    (r"from qpocket_pipeline import (api|tasks|slurm)((?:, (?:api|tasks|slurm))*)\b(?!,)",
     r"from qpocket.pipelines import \1\2"),
    (r"from qpocket_pipeline import api, utils", "from qpocket import utils\nfrom qpocket.pipelines import api"),
    (r"qpocket_pipeline[./]distance_logic([./])", r"qpocket\1core\1"),
    (r"qpocket_pipeline[./]processing([./])", r"qpocket\1core\1"),
    (r"qpocket_pipeline\.distance_logic\b", "qpocket.core"),
    (r"qpocket_pipeline\.processing\b", "qpocket.core"),
    (r"qpocket_pipeline([./])(api|tasks|slurm)\b", r"qpocket\1pipelines\1\2"),
    (r"qpocket_pipeline([./])global_vars\b", r"qpocket\1constants"),
    (r"qpocket_screen([./])", r"qpocket\1screen\1"),
    (r"(?<![\w=-])qpocket_screen\b(?!\.)", "qpocket.screen"),
    # The run log keeps its name: it is a pipeline output, not an import.
    (r"(?<![\w-])qpocket_pipeline\b(?!\.log)", "qpocket"),
]


def rewrite(text: str) -> str:
    """Apply every mapping in order.

    Args:
        text: File contents.

    Returns:
        Rewritten contents.
    """
    for old, new in MAP:
        text = re.sub(old, new, text)
    return text


if __name__ == "__main__":
    for path in map(Path, sys.argv[1:]):
        text = path.read_text(encoding="utf-8")
        new = rewrite(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            print("rewrote", path)
