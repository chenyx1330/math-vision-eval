from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from answer_utils import answers_match, extract_answer
from inference_common import create_openai_client, load_api_config, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction jsonl files.")
    parser.add_argument("input", help="Path to prediction jsonl")
    parser.add_argument("--eval-mode", "-m", choices=["naive", "llm"], default="naive")
    parser.add_argument("--config", "-c", default=None, help="Path to api_config.json for llm mode")
    parser.add_argument("--output", "-o", default=None, help="Path to evaluation report json")
    parser.add_argument("--badcases", "-b", default=None, help="Path to badcases jsonl")
    return parser.parse_args()


def derive_dataset_name(input_path: Path) -> str:
    stem = input_path.stem
    return stem.replace("_predictions", "") if "_predictions" in stem else stem


def derive_output_paths(
    input_path: Path,
    eval_mode: str,
    output_path: str | None,
    badcases_path: str | None,
) -> tuple[str, str, str]:
    dataset_name = derive_dataset_name(input_path)
    output_dir = input_path.parent
    report_path = output_path or str(output_dir / f"{dataset_name}_eval_{eval_mode}.json")
    badcases_file = badcases_path or str(output_dir / f"{dataset_name}_badcases_{eval_mode}.jsonl")
    return dataset_name, report_path, badcases_file


def llm_judge_answer(predicted: str, gold: str, client: OpenAI, model: str) -> tuple[bool, str]:
    if not predicted or not gold:
        return False, "empty"

    prompt = (
        "Judge whether the following two answers are semantically equivalent. "
        "Reply with only `equivalent` or `not_equivalent`.\n\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n\n"
        "Treat unit differences, formatting differences, capitalization changes, "
        "and equivalent option wording as equivalent when the meaning is unchanged."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        print(f"LLM evaluation failed, fallback to rule-based matching: {exc}")
        return answers_match(predicted, gold)

    verdict = (response.choices[0].message.content or "").strip().lower()
    if "equivalent" in verdict and "not_equivalent" not in verdict:
        return True, "llm_match"
    return False, "llm_mismatch"


def evaluate_predictions(
    input_file: str,
    eval_mode: str,
    config_path: str | None,
    output_file: str | None,
    badcases_file: str | None,
) -> None:
    input_path = Path(input_file)
    dataset_name, report_path, badcases_path = derive_output_paths(
        input_path=input_path,
        eval_mode=eval_mode,
        output_path=output_file,
        badcases_path=badcases_file,
    )

    client: OpenAI | None = None
    model_name: str | None = None
    if eval_mode == "llm":
        default_config = Path(__file__).resolve().parents[1] / "api_config.json"
        config = load_api_config(Path(config_path) if config_path else default_config)
        client = create_openai_client(config["api_key"], config["base_url"])
        model_name = config["model"]
        print(f"Using llm mode with model: {model_name}")

    predictions = read_jsonl(input_path)
    total = len(predictions)
    correct = 0
    badcases: list[dict] = []

    print(f"Evaluating {total} predictions with mode: {eval_mode}")

    for index, item in enumerate(predictions, start=1):
        gold_answer = str(item.get("answer", "")).strip()
        response_text = str(item.get("response_text", ""))
        predicted_answer = extract_answer(response_text)

        if eval_mode == "llm":
            assert client is not None and model_name is not None
            is_correct, match_type = llm_judge_answer(predicted_answer, gold_answer, client, model_name)
            if index % 10 == 0 or index == total:
                print(f"  processed {index}/{total}")
        else:
            is_correct, match_type = answers_match(predicted_answer, gold_answer)

        if is_correct:
            correct += 1
            continue

        badcases.append(
            {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "match_type": match_type,
                "response_text": response_text,
            }
        )

    accuracy = correct / total if total else 0.0
    report = {
        "dataset": dataset_name,
        "eval_mode": eval_mode,
        "input_file": input_path.name,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "badcases_count": len(badcases),
    }
    if model_name:
        report["model"] = model_name

    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(badcases_path).open("w", encoding="utf-8") as handle:
        for case in badcases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Report: {report_path}")
    print(f"Badcases: {badcases_path}")
    print("=" * 50)


def main() -> None:
    args = parse_args()
    evaluate_predictions(
        input_file=args.input,
        eval_mode=args.eval_mode,
        config_path=args.config,
        output_file=args.output,
        badcases_file=args.badcases,
    )


if __name__ == "__main__":
    main()
