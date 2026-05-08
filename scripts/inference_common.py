from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
from pathlib import Path

from openai import OpenAI
from PIL import Image


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "api_config.json"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful multimodal math reasoning assistant. "
    "Solve the problem step by step. "
    "You must show a detailed reasoning process, then end with one final line exactly as "
    "'Final Answer: <answer>'."
)

MAX_DATA_URI_BYTES = 20 * 1024 * 1024
MAX_IMAGE_FILE_BYTES = 10 * 1024 * 1024
RESIZE_SIDES = [2048, 1600, 1280, 1024]
JPEG_QUALITIES = [85, 75, 65, 55]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_first(sample: dict, *keys: str, default=None):
    for key in keys:
        value = sample.get(key)
        if value not in (None, "", []):
            return value
    return default


def load_api_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    api_key = str(config.get("api_key", "")).strip()
    model = str(config.get("model", "")).strip()
    base_url = str(config.get("base_url", "")).strip()

    if not api_key:
        raise ValueError(f"Missing api_key in config file: {path}")
    if not model:
        raise ValueError(f"Missing model in config file: {path}")
    if not base_url:
        raise ValueError(f"Missing base_url in config file: {path}")

    return {"api_key": api_key, "model": model, "base_url": base_url}


def create_openai_client(api_key: str, base_url: str) -> OpenAI:
    ssl_cert_file = os.environ.pop("SSL_CERT_FILE", None)
    if ssl_cert_file:
        print(f"Removed SSL_CERT_FILE: {ssl_cert_file}")
    return OpenAI(api_key=api_key, base_url=base_url)


def slice_samples(all_samples: list[dict], start_index: int, limit: int | None) -> tuple[int, list[dict]]:
    start_offset = start_index - 1
    if start_offset < 0:
        raise ValueError("--start-index must be at least 1.")
    if start_offset >= len(all_samples):
        raise ValueError(f"--start-index={start_index} is out of range. Total samples: {len(all_samples)}.")

    samples = all_samples[start_offset:]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No samples found.")
    return start_offset, samples


def encode_data_uri(raw_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def should_use_original(raw_bytes: bytes, data_uri: str) -> bool:
    return len(raw_bytes) <= MAX_IMAGE_FILE_BYTES and len(data_uri.encode("utf-8")) <= MAX_DATA_URI_BYTES


def compress_image_if_needed(image_path: Path) -> bytes:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        best_bytes: bytes | None = None
        best_size: int | None = None

        for max_side in RESIZE_SIDES:
            candidate = image.copy()
            longest_side = max(original_width, original_height)
            if longest_side > max_side:
                scale = max_side / longest_side
                new_size = (
                    max(1, int(original_width * scale)),
                    max(1, int(original_height * scale)),
                )
                candidate = candidate.resize(new_size, Image.Resampling.LANCZOS)

            for quality in JPEG_QUALITIES:
                buffer = io.BytesIO()
                candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
                candidate_bytes = buffer.getvalue()
                candidate_data_uri = encode_data_uri(candidate_bytes, "image/jpeg")

                if best_size is None or len(candidate_bytes) < best_size:
                    best_bytes = candidate_bytes
                    best_size = len(candidate_bytes)

                if should_use_original(candidate_bytes, candidate_data_uri):
                    return candidate_bytes

        if best_bytes is None:
            raise ValueError(f"Failed to compress image: {image_path}")
        return best_bytes


def build_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"
    raw_bytes = image_path.read_bytes()
    data_uri = encode_data_uri(raw_bytes, mime_type)
    if should_use_original(raw_bytes, data_uri):
        return data_uri

    optimized_bytes = compress_image_if_needed(image_path)
    optimized_data_uri = encode_data_uri(optimized_bytes, "image/jpeg")
    if not should_use_original(optimized_bytes, optimized_data_uri):
        raise ValueError(f"Image is still too large after compression: {image_path}")
    return optimized_data_uri


def build_user_prompt(sample: dict) -> str:
    question = str(pick_first(sample, "question", "query", "problem", "prompt", default=""))
    parts = [f"Question:\n{question}"]
    choices = pick_first(sample, "options", "choices", default=[]) or []
    if choices:
        parts.append("Options:\n" + "\n".join(str(choice) for choice in choices))
    return "\n\n".join(parts)


def build_multimodal_messages(prompt_text: str, image_url: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def get_sample_fields(sample: dict, dataset_index: int) -> dict:
    sample_id = str(pick_first(sample, "_sample_id", "id", "pid", default=""))
    question = str(pick_first(sample, "question", "query", "problem", "prompt", default=""))
    answer = str(pick_first(sample, "answer", "label", "final_answer", default=""))
    image_path_value = pick_first(sample, "_image_path", default="")
    if not image_path_value:
        raise ValueError(f"Missing _image_path for sample {sample_id or dataset_index}")

    return {
        "id": sample_id,
        "question": question,
        "answer": answer,
        "image_path": Path(str(image_path_value)),
        "prompt": build_user_prompt(sample),
    }


def build_prediction_row(sample_fields: dict, model: str, response_text: str, latency_sec: float, **extra) -> dict:
    row = {
        "id": sample_fields["id"],
        "question": sample_fields["question"],
        "answer": sample_fields["answer"],
        "image_path": str(sample_fields["image_path"]),
        "model": model,
        "response_text": response_text,
        "latency_sec": round(latency_sec, 3),
    }
    row.update(extra)
    return row
