from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run batch multimodal inference through an OpenAI-compatible API.")
    parser.add_argument("--input", required=True, help="Path to records.jsonl")
    parser.add_argument("--output", required=True, help="Path to prediction jsonl")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to api_config.json")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based sample index to start from. For example, 745 means start from the 745th record.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    parser.add_argument("--append", action="store_true", help="Append predictions to the output file instead of overwriting it")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
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

            response = client.chat.completions.create(
                model=config["model"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                messages=build_multimodal_messages(sample_fields["prompt"], build_data_uri(sample_fields["image_path"])),
            )

            row = build_prediction_row(
                sample_fields=sample_fields,
                model=config["model"],
                response_text=response.choices[0].message.content or "",
                latency_sec=time.time() - started_at,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed_in_run = idx - start_offset
            print(
                f"[{processed_in_run}/{len(samples)}] finished sample {sample_fields['id'] or idx} "
                f"(dataset index {idx})"
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
