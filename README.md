# TestHallVQA

Official repository for the paper **“TestHallVQA: Exploring LVLMs' Document-Level Reasoning under Redundant Contexts from Scientific Exams.”**

> **Complete theoretical derivation:** The full theoretical derivation referenced in the paper is provided in [Complete Theoretical Derivation.pdf](./Complete%20Theoretical%20Derivation.pdf). This document contains the complete derivation beyond the condensed presentation in the main paper.

TestHallVQA evaluates whether large vision-language models (LVLMs) can answer questions from scientific examination papers when the relevant evidence is surrounded by increasingly long redundant visual context. Each benchmark record provides a question, one or more gold-answer components, answer-relevant pages, and several optional context levels containing additional document pages.

Given a selected page field, a model receives the corresponding page images in the stored order together with the prompt:

```text
Please Answer Question {question}
```

## Dataset at a glance

| Item | Count |
| --- | ---: |
| Questions | 10,242 |
| Examination documents | 382 |
| Page images | 7,155 |
| Gold-answer components | 11,317 |
| Subjects | Chemistry, Mathematics, Physics |

### Questions by subject

| Subject | Questions |
| --- | ---: |
| Chemistry | 3,709 |
| Mathematics | 3,302 |
| Physics | 3,231 |

### Gold-answer components by type

One question may contain multiple gold-answer components. Therefore, the component counts below sum to 11,317 rather than the number of questions.

| Type | Components |
| --- | ---: |
| `normal` | 4,189 |
| `calculate` | 2,755 |
| `choice` | 1,923 |
| `state` | 1,940 |
| `show` | 410 |
| `judge` | 100 |

## Repository contents

```text
TestHallVQA-benchmark/
├── TestHallVQA.jsonl                 # 10,242 benchmark annotations
├── images/                           # 7,155 page images stored with Git LFS
├── evaluation.py                     # Model-agnostic inference example
├── score.py                          # Reference judge-based scorer and prompts
├── Complete Theoretical Derivation.pdf
└── README.md
```

## Download

The PNG files are tracked with Git LFS. Install Git LFS before cloning so that the images are downloaded rather than left as pointer files.

```bash
git lfs install
git clone git@github.com:yqyu2317/TestHallVQA-benchmark.git
cd TestHallVQA-benchmark
git lfs pull
```

HTTPS cloning also works:

```bash
git clone https://github.com/yqyu2317/TestHallVQA-benchmark.git
```

## JSONL format

`TestHallVQA.jsonl` contains one JSON object per line. A representative record has the following structure:

```json
{
  "exam_id": "AQA_A-LEVEL__Chemistry_(7405)__Chemistry_1_Summer_2017",
  "question": "01.1",
  "solution": "",
  "answer": [
    "Enthalpy change or heat energy change when 1 mol of solid ionic compound is formed from its gaseous ions."
  ],
  "type": ["state"],
  "subject": "Chemistry",
  "all_page_num": 18,
  "answer_pages": [0],
  "level_1_pages": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "level_2_pages": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
  "levels": [1, 10, 18]
}
```

### Field definitions

| Field | Description |
| --- | --- |
| `exam_id` | Unique identifier shared by all questions and page images from one examination document. |
| `question` | Question identifier printed on the examination paper. |
| `solution` | Reference solution or reasoning when available; it may be empty. |
| `answer` | List of gold-answer components. |
| `type` | Scoring type for each corresponding element in `answer`; the two lists always have equal length. |
| `subject` | `Chemistry`, `Math`, or `Physics`. |
| `all_page_num` | Total number of available page images for the examination document. |
| `answer_pages` | Minimal answer-relevant page indices. |
| `level_1_pages` ... `level_4_pages` | Increasing visual-context levels that retain the answer pages and add redundant pages. Not every record has every level. |
| `levels` | Number of images at each available context setting, in the same order as `answer_pages`, `level_1_pages`, ..., `level_4_pages`. |

Page indices are zero-based and must be used in the order stored in the selected list.

## Visual-context levels

`answer_pages` is available for every question. Higher context levels are included only when the examination document supports the corresponding context size.

| Page field | Available questions | Images per question |
| --- | ---: | ---: |
| `answer_pages` | 10,242 | 1–4 |
| `level_1_pages` | 9,327 | 6–10 |
| `level_2_pages` | 7,977 | 15–20 |
| `level_3_pages` | 3,772 | 25–30 |
| `level_4_pages` | 859 | 35–40 |

All available `level_*_pages` lists contain the corresponding `answer_pages`; the additional images provide redundant document context.

## Resolving page images

Every image is stored directly under `images/` and follows this naming rule:

```text
{exam_id}_{page_index}.png
```

For example, page `0` of the record above is:

```text
images/AQA_A-LEVEL__Chemistry_(7405)__Chemistry_1_Summer_2017_0.png
```

The core input construction is therefore:

```python
from pathlib import Path

pages_field = "level_2_pages"
image_paths = [
    Path("images") / f"{record['exam_id']}_{page}.png"
    for page in record[pages_field]
]
prompt = f"Please Answer Question {record['question']}"
```

Do not sort the resolved paths independently: the order in the selected page list is the model-input order.

## Running model inference

[`evaluation.py`](./evaluation.py) provides a model-agnostic reference pipeline. Implement a `generate` callback for the LVLM or serving framework being evaluated:

```python
from evaluation import evaluate_jsonl


def generate(images, prompt, generation_config):
    """Return one model response for the ordered images and prompt."""
    # Load/process the images and call your multimodal model here.
    # Map max_tokens to the equivalent argument used by your backend.
    return model_response


summary = evaluate_jsonl(
    dataset_path="TestHallVQA.jsonl",
    images_dir="images",
    output_path="predictions_answer_pages.jsonl",
    pages_field="answer_pages",
    generate=generate,
)
print(summary)
```

Use one of the following page fields as the evaluation condition:

```text
answer_pages
level_1_pages
level_2_pages
level_3_pages
level_4_pages
```

When evaluating a higher-level subset, set `skip_unavailable=True` to skip records that do not contain the selected field. Otherwise, a missing field raises an error.

### Reference generation settings

| Parameter | Value |
| --- | ---: |
| `temperature` | `0.0` |
| `repetition_penalty` | `1.05` |
| `max_tokens` | `2000` |

The output JSONL preserves the original annotation and adds:

| Output field | Description |
| --- | --- |
| `prediction` | Full text returned by the evaluated model. |
| `evaluation_pages_field` | Page field used to construct that model input. |

## Scoring

[`score.py`](./score.py) contains the official judge prompts and a model-agnostic scoring implementation. Supply a `judge(prompt) -> str` callback backed by the judge model or API of your choice.

The reference judge is:

```text
Qwen/Qwen3-30B-A3B-Instruct-2507
```

Its reference generation settings are `temperature=0.0`, `repetition_penalty=1.05`, and `max_tokens=2048`.

The scorer creates one judge request for every paired element of `answer` and `type`, then averages the component scores to obtain the question score.

```python
from score import evaluate_record


def judge(prompt):
    """Send prompt as a user message and return the judge model response."""
    return judge_response


result = evaluate_record(
    annotation=record,
    predicted_answer=record["prediction"],
    judge=judge,
)
print(result["score"])
```

If the evaluated model provides separate reasoning, pass it as `predicted_reasoning`. Otherwise, the prediction is used as both the final answer and the reasoning for `show` and `judge` questions.

### Scoring rules

| Type | Component score |
| --- | --- |
| `normal` | `1` for a semantically correct/equivalent answer, otherwise `0`. |
| `choice` | `1` only when the selected option or option content matches exactly, otherwise `0`. |
| `calculate` | Numerical pairs are checked with a 5% relative-error tolerance; pair scores are averaged. |
| `state` | `1` when the required points are covered, `0.5` for insufficient but partially correct coverage, otherwise `0`. |
| `show` | `1` for an aligned proof approach, `0.5` for a close but partially deviating approach, otherwise `0`. |
| `judge` | `1` for the correct judgment and reasoning, `0.5` for the correct judgment with incorrect reasoning, otherwise `0`. |

Malformed or unparseable judge output receives a component score of `0`.

## Complete theoretical derivation

The paper refers to a complete theoretical derivation that is provided separately in this repository. See [Complete Theoretical Derivation.pdf](./Complete%20Theoretical%20Derivation.pdf) for the full derivation.

## Citation

If you use TestHallVQA in your research, please cite:

> **TestHallVQA: Exploring LVLMs' Document-Level Reasoning under Redundant Contexts from Scientific Exams.**

Full citation metadata will be added with the paper release.
