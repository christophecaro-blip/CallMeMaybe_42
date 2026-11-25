"""Read the input files and write the output file."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.errors import InputFileError
from src.models import FunctionDefinition, PromptItem, FunctionCall

Item = TypeVar("Item", bound=BaseModel)


def _read_json_list(path: Path, what: str) -> list[Any]:
    """Read a JSON file that must contain a non-empty list.

    Args:
        path: file to read.
        what: plural name of the items ("functions", "prompts"),
            used in error messages.

    Raises:
        InputFileError: if the file is unreadable, not UTF-8 text, not
            valid JSON, not a list, or an empty list.
    """
    try:
        with open(path, "r", encoding="utf-8") as ft:
            data = json.load(ft)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InputFileError(f"{path}: {e}") from e

    if not isinstance(data, list):
        raise InputFileError(
            f"{path}: expected a list of {what}, "
            f"got {type(data).__name__}"
        )
    if not data:
        raise InputFileError(f"{path}: no {what} defined (empty list)")
    return data


def _summarize(error: ValidationError) -> str:
    """Turn a pydantic error into one readable line, without its URL.

    Example: "parameters.a.type: Input should be 'number', 'string'...".
    """
    parts = []
    for detail in error.errors():
        location = ".".join(str(key) for key in detail["loc"]) or "item"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)


def _validate_items(
    data: list[Any], model: type[Item], path: Path, label: str
) -> list[Item]:
    """Validate each element, naming the faulty one in the error.

    Args:
        data: elements read from the file.
        model: pydantic model every element must match.
        path: file the elements come from (for the message).
        label: singular name of an element ("function", "prompt").

    Raises:
        InputFileError: an element does not match, e.g.
            "tests.json: prompt #2: prompt: Field required".
    """
    items = []
    for number, element in enumerate(data, start=1):
        try:
            items.append(model.model_validate(element))
        except ValidationError as e:
            raise InputFileError(
                f"{path}: {label} #{number}: {_summarize(e)}") from e
    return items


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions from a JSON file.

    Raises:
        InputFileError: if the file is unreadable, not valid JSON,
            or does not match the expected schema.
    """
    data = _read_json_list(path, "functions")
    return _validate_items(data, FunctionDefinition, path, "function")


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the prompts from a JSON file.

    Raises:
        InputFileError: if the file is unreadable, not valid JSON,
            or does not match the expected schema.
    """
    data = _read_json_list(path, "prompts")
    return _validate_items(data, PromptItem, path, "prompt")


def write_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write the function calls to a JSON file, creating parent folders.

    Raises:
        InputFileError: if the folder or the file cannot be written.
    """
    data = [call.model_dump() for call in calls]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as ft:
            json.dump(data, ft, indent=2, ensure_ascii=False)
    except OSError as e:
        raise InputFileError(f"{path}: {e}") from e
