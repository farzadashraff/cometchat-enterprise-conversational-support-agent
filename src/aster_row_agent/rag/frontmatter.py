"""YAML front-matter parsing for the supplied Markdown documents.

Parsing never mutates the source text — it only splits an in-memory copy of
the file content into (front_matter_dict, body_text) and returns both. The
original file on disk is never opened for writing by any part of this
package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aster_row_agent.rag.errors import FrontMatterError

_DELIMITER = "---"


def split_front_matter(raw_text: str, *, path: Path) -> tuple[dict[str, Any], str]:
    """Split raw Markdown text into (front_matter, body).

    Expects the standard format:

        ---
        key: value
        ---
        # Body starts here

    Raises :class:`FrontMatterError` if the document does not open with a
    front-matter block or the YAML between the delimiters does not parse to
    a mapping. This is a deliberate hard failure ("fail clearly if ingestion
    encounters an invalid source") rather than silently treating the whole
    file as body text.
    """
    lines = raw_text.split("\n")
    if not lines or lines[0].strip() != _DELIMITER:
        raise FrontMatterError(path, "document does not start with a '---' front-matter delimiter")

    try:
        closing_index = next(
            index for index in range(1, len(lines)) if lines[index].strip() == _DELIMITER
        )
    except StopIteration as exc:
        raise FrontMatterError(path, "no closing '---' front-matter delimiter found") from exc

    yaml_block = "\n".join(lines[1:closing_index])
    try:
        parsed = yaml.safe_load(yaml_block)
    except yaml.YAMLError as exc:
        raise FrontMatterError(path, f"front matter is not valid YAML: {exc}") from exc

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise FrontMatterError(path, "front matter must parse to a YAML mapping")

    body = "\n".join(lines[closing_index + 1 :])
    return parsed, body
