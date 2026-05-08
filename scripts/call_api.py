from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_common import (
    DEFAULT_CONFIG_PATH,
    build_data_uri,
    build_multimodal_messages,
    create_openai_client,
    get_sample_fields,
    load_api_config,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a multimodal API on one downloaded sample.")
    parser.add_argument("--input", required=True, help="Path to records.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional output json path")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to api_config.json",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    config = load_api_config(Path(args.config))
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No samples found in {input_path}")
    if args.sample_index < 0 or args.sample_index >= len(rows):
        raise IndexError(f"sample-index out of range: {args.sample_index}")

    sample_fields = get_sample_fields(rows[args.sample_index], args.sample_index + 1)
    client = create_openai_client(config["api_key"], config["base_url"])
    response = client.chat.completions.create(
        model=config["model"],
        temperature=0,
        max_tokens=1024,
        messages=build_multimodal_messages(sample_fields["prompt"], build_data_uri(sample_fields["image_path"])),
    )

    result = {
        "model": config["model"],
        "question": sample_fields["question"],
        "image_path": str(sample_fields["image_path"]),
        "response_text": response.choices[0].message.content or "",
    }

    output_path = Path(args.output) if args.output else input_path.parent / "api_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
