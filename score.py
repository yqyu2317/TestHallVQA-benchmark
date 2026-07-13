"""Reference evaluator for TestHallVQA.

This file documents the official scoring procedure without prescribing a
specific inference framework.  A prediction is evaluated against every
``(answer, type)`` component in one TestHallVQA annotation.  The component
scores are then averaged to obtain the question score.

The caller only needs to provide a ``judge(prompt) -> str`` function backed by
the language model or API of their choice.  Send each prompt as a user message
and return the generated text.

Example::

    result = evaluate_record(
        annotation=record_from_testhallvqa,
        predicted_answer="The model's final answer",
        predicted_reasoning="The model's full response or reasoning",
        judge=call_your_judge_model,
    )
    print(result["score"])

``predicted_reasoning`` is used for ``show`` and ``judge`` questions because
those types require evaluating the reasoning process.  The final answer is
used for all other question types.  If no separate reasoning is available,
omit it and the final answer will be used for both.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any


Judge = Callable[[str], str]

# Reference setup used to produce the benchmark scores.  These values are
# metadata for callers; this module intentionally does not initialize a model.
REFERENCE_JUDGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REFERENCE_GENERATION_CONFIG = {
    "temperature": 0.1,
    "repetition_penalty": 1.05,
    "max_tokens": 2048,
}


PROMPT_TEMPLATES = {
    "normal": """You will act as an examiner, comparing the candidate's answer with the standard answer and giving a score. Judge whether the standard answer is present inside the candidate's answer, allowing for formatting differences such as words vs. symbols, fractions vs. decimals, or equivalent expressions. The standard answer tends to be more general and basic; the candidate's answer earns credit if it satisfies the constraints defined by the gold standard. However, if the candidate's answer contains only a limited part of the standard answer, it should be strictly marked as 0. Award 1 point if the answer is correct and 0 points otherwise.

Candidate's answer: {candidate_answer}

Standard answer: {gold_answer}

Analyze the comparison, then end with exactly one line in this format:
Final score: <0 or 1>""",

    "show": """You will act as an examiner, comparing the candidate's answer with the standard answer and giving a rational score. This is a proof question, so determine whether the candidate's proof process aligns with the standard proof process. Differences in phrasing are acceptable, but the core logical steps and backbone of the reasoning must remain consistent. Award 1 point if the general approach is similar, 0.5 points if the approach deviates slightly but is still reasonably close, and 0 points if the proof approach is unrelated to the standard proof process.

Candidate's answer and reasoning: {candidate_reasoning}

Standard answer: {gold_answer}

Analyze the comparison, then end with exactly one line in this format:
Final score: <0, 0.5, or 1>""",

    "calculate": """You are an examiner. Follow these steps to compare the candidate's predicted answer with the gold-standard answer:

1. Pairing (one-to-one mapping): Analyze all numerical results in the gold-standard answer. For each number, find the most semantically matching and numerically similar value in the candidate's answer. Each candidate value may be used at most once. If a gold value has no corresponding prediction, do not create a pair for it.
2. Normalization: For each pair, normalize both values to decimal multiplied by the same power of ten.
3. Function calls: For every pair, return one call in the form ``CALCULATE_ERROR(gold_number, predicted_number)``. Use ordinary decimal or scientific notation and do not perform the calculation yourself.

Candidate's answer: {candidate_answer}

Gold-standard answer: {gold_answer}

Return only the CALCULATE_ERROR calls, one per line.""",

    "state": """You will act as an examiner, comparing the candidate's answer with the standard answer and giving a score. This is a discussion question.

Step 1 - Required number of matches: Check whether the standard answer requires at least a certain number of points, for example "any four of". If it does, the candidate must provide at least that many correct matches. If no minimum is specified, one correct match is sufficient.

Step 2 - Correct matches: A candidate point is a correct match when it is semantically consistent with, or conveys a meaning similar to, a point in the standard answer.

Step 3 - Score:
- Award 1 point if correct matches exist and the minimum-number requirement is satisfied.
- Award 0.5 points if some correct matches exist but the required number is not reached.
- Award 0 points if there is no correct match.

Candidate's answer: {candidate_answer}

Standard answer: {gold_answer}

Analyze the comparison, then end with exactly one line in this format:
Final score: <0, 0.5, or 1>""",

    "judge": """You will act as an examiner, comparing the candidate's answer with the standard answer and giving a score. This is a judgment question.

- Award 0 points if the candidate's judgment is opposite to the standard answer.
- If the judgment is correct, award 1 point when the reasoning agrees with the standard reasoning.
- Award 0.5 points when the judgment is correct but the reasoning is incorrect or inconsistent with the standard reasoning.

Candidate's answer and reasoning: {candidate_reasoning}

Standard answer: {gold_answer}
Standard solution: {gold_solution}

Analyze the comparison, then end with exactly one line in this format:
Final score: <0, 0.5, or 1>""",

    "choice": """You will act as an examiner, comparing the candidate's answer with the standard answer and giving a score. Determine whether the option selected by the candidate matches the standard answer. Accept equivalent option representations, including labels such as A/B/C/D, ordinal descriptions such as "the second option", or the corresponding option text. For multiple-choice questions, the selected options must match the standard answer exactly; partial matches or omissions receive 0 points. Award 1 point for a correct answer and 0 points otherwise.

Candidate's answer: {candidate_answer}

Standard answer: {gold_answer}

Analyze the comparison, then end with exactly one line in this format:
Final score: <0 or 1>""",
}


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CALCULATE_CALL_RE = re.compile(
    rf"CALCULATE_ERROR\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)",
    re.IGNORECASE,
)
_FINAL_SCORE_RE = re.compile(
    r"final\s+score\s*[:=]\s*(0(?:\.5)?|1(?:\.0)?)\b",
    re.IGNORECASE,
)
_SCORE_FALLBACK_RE = re.compile(
    r"(?<![\d.])(0(?:\.5)?|1(?:\.0)?)(?![\d.])"
)


def calculate_error(gold_number: float, predicted_number: float) -> float:
    """Return 1 when the prediction is within 5% of the gold value."""

    if gold_number == 0:
        return 1.0 if predicted_number == 0 else 0.0
    relative_error = abs(predicted_number - gold_number) / abs(gold_number)
    return 1.0 if relative_error <= 0.05 else 0.0


def build_prompt(
    question_type: str,
    predicted_answer: str,
    predicted_reasoning: str,
    gold_answer: str,
    gold_solution: str = "",
) -> str:
    """Build the official judge prompt for one answer component."""

    try:
        template = PROMPT_TEMPLATES[question_type]
    except KeyError as exc:
        supported = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unsupported question type {question_type!r}; expected one of: {supported}"
        ) from exc

    return template.format(
        candidate_answer=predicted_answer,
        candidate_reasoning=predicted_reasoning,
        gold_answer=gold_answer,
        gold_solution=gold_solution,
    )


def parse_judge_output(question_type: str, output: str) -> float:
    """Convert one judge response into a component score.

    A malformed response receives 0, which keeps the evaluation deterministic
    and guarantees that every component contributes to the final average.
    """

    if question_type == "calculate":
        pairs = _CALCULATE_CALL_RE.findall(output.replace("−", "-"))
        if not pairs:
            return 0.0
        scores = [
            calculate_error(float(gold), float(predicted))
            for gold, predicted in pairs
        ]
        return sum(scores) / len(scores)

    matches = _FINAL_SCORE_RE.findall(output)
    if not matches:
        # Compatibility fallback for judges that return the score as the last
        # standalone number but do not follow the requested final-line format.
        matches = _SCORE_FALLBACK_RE.findall(output)
    return float(matches[-1]) if matches else 0.0


def build_evaluation_requests(
    annotation: Mapping[str, Any],
    predicted_answer: str,
    predicted_reasoning: str | None = None,
) -> list[dict[str, str]]:
    """Create one judge request for each gold-answer component."""

    answers = annotation.get("answer")
    question_types = annotation.get("type")
    if not isinstance(answers, list) or not isinstance(question_types, list):
        raise TypeError("annotation must contain list-valued 'answer' and 'type' fields")
    if not answers or len(answers) != len(question_types):
        raise ValueError("'answer' and 'type' must be non-empty lists of equal length")

    reasoning = predicted_answer if predicted_reasoning is None else predicted_reasoning
    solution = str(annotation.get("solution", ""))
    requests = []

    for question_type, gold_answer in zip(question_types, answers):
        question_type = str(question_type)
        prompt = build_prompt(
            question_type=question_type,
            predicted_answer=str(predicted_answer),
            predicted_reasoning=str(reasoning),
            gold_answer=str(gold_answer),
            gold_solution=solution,
        )
        requests.append({"type": question_type, "prompt": prompt})

    return requests


def evaluate_record(
    annotation: Mapping[str, Any],
    predicted_answer: str,
    judge: Judge,
    predicted_reasoning: str | None = None,
) -> dict[str, Any]:
    """Evaluate one TestHallVQA record and return its averaged score.

    ``judge`` should send the supplied prompt to the chosen judge model as a
    user message and return the model's generated text.  For efficient dataset
    evaluation, callers may batch the requests produced by
    :func:`build_evaluation_requests` and apply :func:`parse_judge_output` to
    the returned texts.
    """

    requests = build_evaluation_requests(
        annotation=annotation,
        predicted_answer=predicted_answer,
        predicted_reasoning=predicted_reasoning,
    )

    component_scores = []
    judge_outputs = []
    for request in requests:
        output = judge(request["prompt"])
        if not isinstance(output, str):
            raise TypeError("judge(prompt) must return a string")
        judge_outputs.append(output)
        component_scores.append(parse_judge_output(request["type"], output))

    return {
        "score": sum(component_scores) / len(component_scores),
        "component_scores": component_scores,
        "judge_outputs": judge_outputs,
    }
