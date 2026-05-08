#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run self-consistency inference and vote on the final answer."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

from answer_utils import clean_answer, extract_answer
from inference_common import (
    DEFAULT_CONFIG_PATH,
    build_data_uri,
    build_multimodal_messages,
    build_prediction_row,
    create_openai_client,
    get_sample_fields,
    load_api_config,
    read_jsonl,
    slice_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run self-consistency inference through an OpenAI-compatible API."
    )
    parser.add_argument("--input", required=True, help="Path to records.jsonl")
    parser.add_argument("--output", required=True, help="Path to prediction jsonl")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to api_config.json")
    parser.add_argument("--start-index", type=int, default=1, help="1-based sample index to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    parser.add_argument("--append", action="store_true", help="Append predictions to the output file instead of overwriting it")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between samples")
    parser.add_argument(
        "--num-samples",
        "--n",
        dest="num_samples",
        type=int,
        default=4,
        help="Number of candidate responses per sample. Default is 4.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature. Must be > 0.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per response")
    return parser.parse_args()


def generate_candidates(
    client: OpenAI,
    config: dict,
    base_messages: list[dict],
    num_samples: int,
    temperature: float,
    max_tokens: int,
) -> list[dict]:
    response = client.chat.completions.create(
        model=config["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
        messages=base_messages,
    )

    candidates: list[dict] = []
    for attempt_index, choice in enumerate(response.choices, start=1):
        response_text = choice.message.content or ""
        extracted_answer = extract_answer(response_text)
        candidates.append(
            {
                "attempt_index": attempt_index,
                "response_text": response_text,
                "extracted_answer": extracted_answer,
                "normalized_answer": clean_answer(extracted_answer) if extracted_answer else "",
            }
        )
    return candidates


def vote_answer(candidates: list[dict]) -> tuple[str, str, dict]:
    valid_candidates = [candidate for candidate in candidates if candidate["extracted_answer"]]

    print("  Candidate Final Answer:")
    for candidate in candidates:
        extracted = candidate["extracted_answer"] or "[not extracted]"
        print(f"    - response {candidate['attempt_index']}: {extracted}")

    if not valid_candidates:
        return "", candidates[0]["response_text"] if candidates else "", {
            "num_samples": len(candidates),
            "final_answer": "",
            "confidence": 0.0,
            "vote_counts": {},
            "candidates": candidates,
        }

    vote_counts = Counter(candidate["normalized_answer"] for candidate in valid_candidates)
    final_answer, max_votes = vote_counts.most_common(1)[0]
    best_candidate = next(
        candidate for candidate in valid_candidates if candidate["normalized_answer"] == final_answer
    )

    metadata = {
        "num_samples": len(candidates),
        "final_answer": final_answer,
        "confidence": max_votes / len(valid_candidates),
        "vote_counts": dict(vote_counts),
        "candidates": candidates,
    }
    return final_answer, best_candidate["response_text"], metadata


def main() -> None:
    args = parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0 for self-consistency.")

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
            started_at = time.time()
            messages = build_multimodal_messages(
                sample_fields["prompt"],
                build_data_uri(sample_fields["image_path"]),
            )

            processed_in_run = idx - start_offset
            print(
                f"[{processed_in_run}/{len(samples)}] generating {args.num_samples} candidates "
                f"for sample {sample_fields['id'] or idx}..."
            )

            candidates = generate_candidates(
                client=client,
                config=config,
                base_messages=messages,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            final_answer, best_response, metadata = vote_answer(candidates)

            print(f"  -> final answer: {final_answer} (confidence {metadata['confidence']:.2f})")

            row = build_prediction_row(
                sample_fields=sample_fields,
                model=config["model"],
                response_text=best_response,
                latency_sec=time.time() - started_at,
                sc_metadata=metadata,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
