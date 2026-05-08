from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from answer_utils import extract_answer
from inference_common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SYSTEM_PROMPT,
    build_data_uri,
    build_prediction_row,
    create_openai_client,
    load_api_config,
    pick_first,
    read_jsonl,
    slice_samples,
    get_sample_fields,
    build_user_prompt,
)


ATTEMPT_INSTRUCTIONS = [
    "Solve the problem carefully step by step and end with 'Final Answer: <answer>'.",
    (
        "Your previous attempt had an inconsistency between the reasoning process and the final answer. "
        "Re-solve the problem from the beginning. Before writing the final answer, explicitly check that "
        "the last calculation or selected option is exactly the same as the final answer."
    ),
    (
        "Your previous attempt still had an inconsistency. Re-solve the problem again with stricter checking. "
        "You must verify the visual details, re-check the final calculation or option choice, and ensure the "
        "final answer is fully consistent with the reasoning. Do not copy the previous mistake."
    ),
]

JUDGE_SYSTEM_PROMPT = (
    "You are a strict verifier for multimodal math reasoning. "
    "Your job is not to solve the full problem again. "
    "Only check whether the reasoning process is consistent with the stated final answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative generation -> verification -> retry inference through an OpenAI-compatible API."
    )
    parser.add_argument("--input", required=True, help="Path to records.jsonl")
    parser.add_argument("--output", required=True, help="Path to prediction jsonl")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to api_config.json")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based sample index to start from.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    parser.add_argument("--append", action="store_true", help="Append predictions to the output file")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between samples")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generator temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for generation")
    parser.add_argument("--judge-max-tokens", type=int, default=256, help="Max tokens for judge output")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum generation attempts per sample")
    return parser.parse_args()


def build_generator_messages(
    sample: dict,
    image_url: str,
    attempt_index: int,
    previous_response: str = "",
    judge_feedback: str = "",
) -> list[dict]:
    user_text = build_user_prompt(sample)
    extra_instruction = ATTEMPT_INSTRUCTIONS[min(attempt_index, len(ATTEMPT_INSTRUCTIONS) - 1)]

    text_parts = [user_text, f"Additional instruction:\n{extra_instruction}"]

    if previous_response and judge_feedback:
        text_parts.append(
            "Previous attempt:\n"
            f"{previous_response}\n\n"
            "Verifier feedback about the previous attempt:\n"
            f"{judge_feedback}"
        )

    text_parts.append("Remember: the final line must be exactly in the format 'Final Answer: <answer>'.")

    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "\n\n".join(text_parts)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def build_judge_messages(sample: dict, response_text: str) -> list[dict]:
    question = str(pick_first(sample, "question", "query", "problem", "prompt", default=""))
    choices = pick_first(sample, "options", "choices", default=[]) or []
    choice_block = ""
    if choices:
        choice_block = "Options:\n" + "\n".join(str(choice) for choice in choices) + "\n\n"

    extracted_final = extract_answer(response_text)

    judge_prompt = (
        "You will receive a math question, the model's reasoning process, and its stated final answer.\n"
        "Check only whether the reasoning is consistent with the stated final answer.\n"
        "If the answer is inconsistent with the reasoning, suggest a more credible corrected final answer.\n\n"
        f"Question:\n{question}\n\n"
        f"{choice_block}"
        f"Model reasoning and answer:\n{response_text}\n\n"
        f"Extracted final answer:\n{extracted_final}\n\n"
        "Output only a JSON object with the following keys:\n"
        "{\n"
        '  "consistent": true or false,\n'
        '  "reason": "short explanation",\n'
        '  "corrected_final_answer": "answer string, or empty string if consistent",\n'
        '  "confidence": 0 to 1\n'
        "}\n"
        "Do not add any extra text outside the JSON object."
    )

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]


def parse_json_object(text: str) -> dict:
    text = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Judge output is not a JSON object.")
    return data


def normalize_judge_result(raw: dict, fallback_answer: str) -> dict:
    consistent = raw.get("consistent", False)
    if isinstance(consistent, str):
        consistent = consistent.strip().lower() in {"true", "yes", "y", "1"}
    consistent = bool(consistent)

    reason = str(raw.get("reason", "")).strip()
    corrected = str(raw.get("corrected_final_answer", "")).strip()
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "consistent": consistent,
        "reason": reason,
        "corrected_final_answer": corrected,
        "confidence": confidence,
        "extracted_final_answer": fallback_answer,
    }


def judge_response(
    client: OpenAI,
    config: dict,
    sample: dict,
    response_text: str,
    max_tokens: int,
) -> dict:
    fallback_answer = extract_answer(response_text)
    response = client.chat.completions.create(
        model=config["model"],
        temperature=0.0,
        max_tokens=max_tokens,
        messages=build_judge_messages(sample, response_text),
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_object(content)
    result = normalize_judge_result(parsed, fallback_answer)
    result["raw_judge_text"] = content
    return result


def run_single_sample(
    client: OpenAI,
    config: dict,
    sample: dict,
    image_url: str,
    temperature: float,
    max_tokens: int,
    judge_max_tokens: int,
    max_attempts: int,
) -> tuple[str, dict]:
    previous_response = ""
    previous_feedback = ""
    attempts: list[dict] = []

    for attempt_index in range(max_attempts):
        messages = build_generator_messages(
            sample=sample,
            image_url=image_url,
            attempt_index=attempt_index,
            previous_response=previous_response,
            judge_feedback=previous_feedback,
        )

        generation = client.chat.completions.create(
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        response_text = generation.choices[0].message.content or ""

        try:
            judge = judge_response(
                client=client,
                config=config,
                sample=sample,
                response_text=response_text,
                max_tokens=judge_max_tokens,
            )
        except Exception as exc:
            judge = {
                "consistent": True,
                "reason": f"judge_failed: {exc}",
                "corrected_final_answer": "",
                "confidence": 0.0,
                "extracted_final_answer": extract_answer(response_text),
                "raw_judge_text": "",
            }

        attempts.append(
            {
                "attempt": attempt_index + 1,
                "attempt_instruction": ATTEMPT_INSTRUCTIONS[min(attempt_index, len(ATTEMPT_INSTRUCTIONS) - 1)],
                "response_text": response_text,
                "judge": judge,
            }
        )

        if judge["consistent"]:
            return response_text, {"attempts": attempts, "accepted_attempt": attempt_index + 1}

        previous_response = response_text
        previous_feedback = (
            f"The previous reasoning and final answer were judged inconsistent.\n"
            f"Reason: {judge['reason']}\n"
            f"Previous extracted final answer: {judge['extracted_final_answer']}\n"
        )
        if judge["corrected_final_answer"]:
            previous_feedback += f"Suggested corrected final answer: {judge['corrected_final_answer']}\n"

    return attempts[-1]["response_text"], {"attempts": attempts, "accepted_attempt": max_attempts}


def main() -> None:
    args = parse_args()

    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")

    config = load_api_config(Path(args.config))
    client = create_openai_client(config["api_key"], config["base_url"])

    all_samples = read_jsonl(Path(args.input))
    start_offset, samples = slice_samples(all_samples, args.start_index, args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.append else "w"

    with output_path.open(output_mode, encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=start_offset + 1):
            sample_fields = get_sample_fields(sample, idx)
            image_url = build_data_uri(sample_fields["image_path"])
            started_at = time.time()

            processed_in_run = idx - start_offset
            print(
                f"[{processed_in_run}/{len(samples)}] processing sample {sample_fields['id'] or idx} "
                f"with max {args.max_attempts} attempts..."
            )

            final_response, retry_metadata = run_single_sample(
                client=client,
                config=config,
                sample=sample,
                image_url=image_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                judge_max_tokens=args.judge_max_tokens,
                max_attempts=args.max_attempts,
            )

            final_extracted = extract_answer(final_response)
            accepted_attempt = retry_metadata.get("accepted_attempt", 1)
            accepted_judge = retry_metadata["attempts"][accepted_attempt - 1]["judge"]

            print(
                f"  -> accepted attempt {accepted_attempt}, consistent={accepted_judge['consistent']}, "
                f"final answer={final_extracted}"
            )

            row = build_prediction_row(
                sample_fields=sample_fields,
                model=config["model"],
                response_text=final_response,
                latency_sec=time.time() - started_at,
                verify_retry_metadata=retry_metadata,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
