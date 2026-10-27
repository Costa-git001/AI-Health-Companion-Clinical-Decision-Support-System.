"""Calculate exact-concept precision, recall and F1 for an annotated JSONL set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.extractor import extract_symptoms  # noqa: E402


def main(path: Path) -> int:
    tp = fp = fn = 0
    rows = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record["text"]
                gold = {str(item).strip().casefold() for item in record["gold_concepts"]}
                if not isinstance(text, str):
                    raise TypeError("text must be a string")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid JSONL row {line_number}: {exc}") from exc
            predicted = {
                str(item["concept"]).casefold()
                for item in extract_symptoms(text)
                if item["status"] != "negated"
            }
            tp += len(predicted & gold)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            rows += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"records: {rows}")
    print(f"true positives: {tp} | false positives: {fp} | false negatives: {fn}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Metric note: exact canonical concept match; aggregate micro scores; no clinical validity inference.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate_extractor.py annotations.jsonl", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
