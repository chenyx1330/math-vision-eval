from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image


DATASET_MAP = {
    "mathvision": "MathLLMs/MathVision",
    "mathvista": "AI4Math/MathVista",
    "mathverse": "AI4Math/MathVerse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one dataset split and save it locally.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_MAP))
    parser.add_argument("--split", default="testmini")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def extract_image(row: dict) -> Image.Image | None:
    for key in ("decoded_image", "image", "images"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            return None
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict) and value.get("bytes"):
            return Image.open(BytesIO(value["bytes"]))
    return None


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Image.Image):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(DATASET_MAP[args.dataset], split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    out_dir = Path(__file__).resolve().parents[1] / "data" / args.dataset / args.split
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            record = to_jsonable(row)
            sample_id = str(record.get("pid") or record.get("id") or f"sample_{index:06d}")
            image = extract_image(row)
            image_path = None
            if image is not None:
                image_path = image_dir / f"{sample_id}.png"
                image.convert("RGB").save(image_path)
            record["_image_path"] = str(image_path) if image_path else None
            record["_sample_id"] = sample_id
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {records_path}")


if __name__ == "__main__":
    main()
