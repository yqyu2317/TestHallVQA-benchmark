"""Reference model-inference pipeline for TestHallVQA.

TestHallVQA is distributed as one JSON object per line plus a flat ``images``
directory. For each record:

1. Choose one page field, such as ``answer_pages`` or ``level_2_pages``.
2. Convert every stored page index into ``{exam_id}_{page_index}.png``.
3. Keep the images in the exact order given by the selected page field.
4. Ask the model ``Please Answer Question {question}``.
5. Save the model response together with the original annotation.

The code below is intentionally model-agnostic. Implement ``generate`` with
the multimodal model or serving framework of your choice. The callback
receives an ordered image-path list, a text prompt, and the reference
generation settings::

    def generate(images, prompt, generation_config):
        # Load/process ``images`` and call your multimodal model here.
        # Map ``max_tokens`` to the equivalent backend argument if necessary.
        return model_response

    evaluate_jsonl(
        dataset_path="TestHallVQA.jsonl",
        images_dir="images",
        output_path="predictions.jsonl",
        pages_field="answer_pages",
        generate=generate,
    )

Use :mod:`score` after inference to compare ``prediction`` with the annotated
answers. This file does not initialize a model, choose a checkpoint path, or
prescribe a particular inference framework.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


PAGE_FIELDS = (
    "answer_pages",
    "level_1_pages",
    "level_2_pages",
    "level_3_pages",
    "level_4_pages",
)

GENERATION_CONFIG: dict[str, float | int] = {
    "temperature": 0.0,
    "repetition_penalty": 1.05,
    "max_tokens": 2000,
}

Generate = Callable[[Sequence[Path], str, Mapping[str, float | int]], str]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON objects from a TestHallVQA JSONL file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def available_page_fields(record: Mapping[str, Any]) -> list[str]:
    """Return the page-context fields provided by one annotation."""

    return [field for field in PAGE_FIELDS if field in record]


def resolve_image_paths(
    record: Mapping[str, Any],
    images_dir: str | Path,
    pages_field: str,
) -> list[Path]:
    """Resolve an annotation's page indices to its ordered PNG paths."""

    if pages_field not in PAGE_FIELDS:
        raise ValueError(
            f"Unsupported pages_field {pages_field!r}; expected one of {PAGE_FIELDS}"
        )
    if pages_field not in record:
        available = ", ".join(available_page_fields(record)) or "none"
        raise KeyError(
            f"Record does not provide {pages_field!r}; available fields: {available}"
        )

    exam_id = record.get("exam_id")
    page_indices = record[pages_field]
    if not isinstance(exam_id, str) or not exam_id:
        raise TypeError("'exam_id' must be a non-empty string")
    if not isinstance(page_indices, list) or not all(
        isinstance(page, int) and not isinstance(page, bool) and page >= 0
        for page in page_indices
    ):
        raise TypeError(f"'{pages_field}' must be a list of non-negative integers")

    images_dir = Path(images_dir)
    image_paths = [images_dir / f"{exam_id}_{page}.png" for page in page_indices]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} image(s) for {exam_id}: {preview}{suffix}"
        )

    return image_paths


def build_question_prompt(record: Mapping[str, Any]) -> str:
    """Build the text instruction paired with the record's page images."""

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise TypeError("'question' must be a non-empty string")
    return f"Please Answer Question {question}"


def prepare_model_input(
    record: Mapping[str, Any],
    images_dir: str | Path,
    pages_field: str,
) -> dict[str, Any]:
    """Create the model-independent input for one benchmark record."""

    return {
        "images": resolve_image_paths(record, images_dir, pages_field),
        "prompt": build_question_prompt(record),
        "generation_config": dict(GENERATION_CONFIG),
    }


def evaluate_record(
    record: Mapping[str, Any],
    images_dir: str | Path,
    pages_field: str,
    generate: Generate,
) -> dict[str, Any]:
    """Run one record and append the prediction without dropping gold fields."""

    model_input = prepare_model_input(record, images_dir, pages_field)
    prediction = generate(
        model_input["images"],
        model_input["prompt"],
        model_input["generation_config"],
    )
    if not isinstance(prediction, str):
        raise TypeError("generate(...) must return the model response as a string")

    result = dict(record)
    result["evaluation_pages_field"] = pages_field
    result["prediction"] = prediction
    return result


def evaluate_jsonl(
    dataset_path: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    pages_field: str,
    generate: Generate,
    *,
    skip_unavailable: bool = False,
) -> dict[str, int]:
    """Evaluate a JSONL file and write one prediction object per output line.

    ``answer_pages`` is present for every TestHallVQA record. Higher-level page
    fields are available only for records that support that context size. By
    default a missing selected field raises an error. Set ``skip_unavailable``
    to ``True`` when intentionally evaluating only the corresponding subset.
    """

    if pages_field not in PAGE_FIELDS:
        raise ValueError(
            f"Unsupported pages_field {pages_field!r}; expected one of {PAGE_FIELDS}"
        )

    output_path = Path(output_path)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as output_stream:
        for record in read_jsonl(dataset_path):
            if pages_field not in record:
                if skip_unavailable:
                    skipped += 1
                    continue
                available = ", ".join(available_page_fields(record)) or "none"
                raise KeyError(
                    f"Record {record.get('exam_id')!r} does not provide "
                    f"{pages_field!r}; available fields: {available}"
                )

            result = evaluate_record(
                record=record,
                images_dir=images_dir,
                pages_field=pages_field,
                generate=generate,
            )
            output_stream.write(json.dumps(result, ensure_ascii=False))
            output_stream.write("\n")
            written += 1

    return {"written": written, "skipped": skipped}
