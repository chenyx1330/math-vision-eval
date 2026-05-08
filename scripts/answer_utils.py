from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


FINAL_ANSWER_PATTERN = re.compile(r"final\s+answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
CHOICE_PATTERN = re.compile(r"^[(\[]?\s*([A-E])\s*[)\].]?$", re.IGNORECASE)
CHOICE_IN_TEXT_PATTERN = re.compile(r"\b(?:option|choice|answer)\s+([A-E])\b", re.IGNORECASE)
EQUATION_NUMBER_PATTERN = re.compile(r"^[a-zA-Z]\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?)$")
UNIT_SUFFIX_PATTERN = re.compile(
    r"\s*(cm|mm|m|km|kg|g|ml|l|ft|feet|inch|inches|yard|yards|degrees?|years?|months?|days?|hours?|minutes?|seconds?)$",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
FRACTION_PATTERN = re.compile(r"^([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)$")
TEXT_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_answer(text: str) -> str:
    if not text or not text.strip():
        return ""

    stripped = text.strip()
    match = FINAL_ANSWER_PATTERN.search(stripped)
    if match:
        return clean_answer(match.group(1).splitlines()[0])

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return clean_answer(lines[-1]) if lines else ""


def clean_answer(text: str) -> str:
    answer = text.strip()
    answer = answer.replace("**", "").replace("`", "")
    answer = re.sub(r"^\$(.+)\$$", r"\1", answer)
    answer = re.sub(r"^\\\((.+)\\\)$", r"\1", answer)

    boxed = re.search(r"\\boxed\{([^}]+)\}", answer)
    if boxed:
        answer = boxed.group(1)

    answer = answer.strip("<>").strip("\"'()[]")
    answer = re.sub(r"^[(\[]?\s*[A-E]\s*[)\].]?\s+", "", answer, flags=re.IGNORECASE)
    answer = answer.rstrip(".,;:!?，。；：！？")
    return answer.strip()


def normalize_answer(text: str) -> str:
    answer = clean_answer(text)
    choice = extract_choice(answer)
    if choice:
        return choice

    number = extract_number(answer)
    if number is not None:
        return number

    return answer.lower()


def extract_choice(text: str) -> str | None:
    stripped = clean_answer(text).strip().upper()
    if stripped in {"A", "B", "C", "D", "E"}:
        return stripped

    match = CHOICE_PATTERN.match(stripped)
    if match:
        return match.group(1).upper()

    match = CHOICE_IN_TEXT_PATTERN.search(stripped)
    if match:
        return match.group(1).upper()
    return None


def extract_number(text: str) -> str | None:
    stripped = clean_answer(text).strip()

    equation_match = EQUATION_NUMBER_PATTERN.match(stripped)
    if equation_match:
        stripped = equation_match.group(1)

    stripped = re.sub(r"^[\$¥£€]\s*", "", stripped)
    stripped = UNIT_SUFFIX_PATTERN.sub("", stripped)

    if stripped.endswith("%"):
        stripped = stripped[:-1].strip()

    stripped = stripped.replace(",", "")

    fraction_match = FRACTION_PATTERN.match(stripped)
    if fraction_match:
        try:
            numerator = Decimal(fraction_match.group(1))
            denominator = Decimal(fraction_match.group(2))
            if denominator != 0:
                return format_decimal(numerator / denominator)
        except InvalidOperation:
            return None

    if NUMBER_PATTERN.match(stripped):
        try:
            return format_decimal(Decimal(stripped))
        except InvalidOperation:
            return None

    return None


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    result = format(normalized, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def extract_all_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    for match in TEXT_NUMBER_PATTERN.findall(clean_answer(text).replace(",", "")):
        try:
            numbers.append(format_decimal(Decimal(match)))
        except InvalidOperation:
            continue
    return numbers


def is_plural_equivalent(left: str, right: str) -> bool:
    if left == right:
        return False
    if left.endswith("s") and left[:-1] == right:
        return True
    if right.endswith("s") and right[:-1] == left:
        return True
    if left.endswith("es") and left[:-2] == right:
        return True
    if right.endswith("es") and right[:-2] == left:
        return True
    return False


def are_synonyms(left: str, right: str) -> bool:
    groups = [
        {"yes", "correct", "true", "right", "正确", "是"},
        {"no", "incorrect", "false", "wrong", "错误", "否"},
    ]
    return any(left in group and right in group for group in groups)


def answers_match(predicted: str, gold: str) -> tuple[bool, str]:
    if not predicted or not gold:
        return False, "empty"

    predicted_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if predicted_norm == gold_norm:
        return True, "exact"

    predicted_num = extract_number(predicted)
    gold_num = extract_number(gold)
    if predicted_num is not None and gold_num is not None:
        try:
            predicted_value = Decimal(predicted_num)
            gold_value = Decimal(gold_num)
            if predicted_value == gold_value:
                return True, "numeric_exact"
            if abs(predicted_value - gold_value) <= Decimal("0.01"):
                return True, "numeric_close"
        except InvalidOperation:
            pass

    if gold_num is not None and predicted_num is None:
        if gold_num in extract_all_numbers(predicted):
            return True, "numeric_contained"

    predicted_choice = extract_choice(predicted)
    gold_choice = extract_choice(gold)
    if predicted_choice and gold_choice and predicted_choice == gold_choice:
        return True, "choice"

    if is_plural_equivalent(predicted_norm, gold_norm):
        return True, "plural"

    if are_synonyms(predicted_norm, gold_norm):
        return True, "synonym"

    if predicted_num is None and gold_num is None and len(predicted_norm) >= 3 and len(gold_norm) >= 3:
        predicted_words = set(predicted_norm.split())
        gold_words = set(gold_norm.split())
        if gold_words and gold_words.issubset(predicted_words):
            return True, "text_contained"

    return False, "mismatch"
