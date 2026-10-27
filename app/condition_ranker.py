"""Closed-label symptom-text benchmark ranker.

This model is trained only for a small, English, LLM-modified benchmark dataset.
Its ranked labels are not diagnoses, clinical probabilities, or advice.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "data" / "train.jsonl"


class ConditionRanker:
    def __init__(self, train_file: Path = TRAIN_FILE) -> None:
        records = self._read_jsonl(train_file)
        if len(records) < 10:
            raise ValueError("Training data is missing or too small.")
        texts = [row["input_text"] for row in records]
        labels = [row["output_text"] for row in records]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
        )
        features = self.vectorizer.fit_transform(texts)
        config_path = PROJECT_ROOT / "data" / "model_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"linear_svc_C": 1.0}
        self.model = LinearSVC(C=float(config.get("linear_svc_C", 1.0)), random_state=42, max_iter=5000)
        self.model.fit(features, labels)
        self.classes = list(self.model.classes_)
        self.feature_names = self.vectorizer.get_feature_names_out()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        rows = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("input_text"), str) or not isinstance(row.get("output_text"), str):
                    raise ValueError(f"Invalid dataset row {line_number}; expected input_text/output_text strings")
                rows.append(row)
        return rows

    def rank(self, text: str, limit: int = 3) -> list[dict[str, object]]:
        vector = self.vectorizer.transform([text])
        decision = self.model.decision_function(vector)[0]
        order = decision.argsort()[::-1][:limit]
        active = vector.toarray()[0]
        results = []
        for position, class_index in enumerate(order, start=1):
            contributions = active * self.model.coef_[class_index]
            top_features = contributions.argsort()[::-1]
            signals = [str(self.feature_names[index]) for index in top_features if contributions[index] > 0][:4]
            results.append({"rank": position, "label": self.classes[class_index], "text_signals": signals})
        return results


_ranker: ConditionRanker | None = None


def get_ranker() -> ConditionRanker:
    global _ranker
    if _ranker is None:
        _ranker = ConditionRanker()
    return _ranker
