"""I/O utility functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

def save_jsonl(data: list[BaseModel], path: str | Path) -> None:
    """Save a list of Pydantic models as a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in data:
            fh.write(item.model_dump_json() + "\n")
    logger.info(f"Saved {len(data)} records to {path}")

def load_jsonl(path: str | Path, model_class: type[ModelT]) -> list[ModelT]:
    """Load a list of Pydantic models from a JSONL file."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []
    items: list[ModelT] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(model_class.model_validate_json(line))
    logger.info(f"Loaded {len(items)} records from {path}")
    return items

def save_json(data: BaseModel | dict, path: str | Path) -> None:
    """Save a single object as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        content = data.model_dump_json(indent=2)
    else:
        content = json.dumps(data, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info(f"Saved to {path}")

def load_json(path: str | Path, model_class: type[ModelT] | None = None) -> ModelT | dict:
    """Load an object from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if model_class:
        return model_class.model_validate(raw)
    return raw