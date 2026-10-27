"""Compare simple text classifiers on the dataset's supplied fixed split."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
RANDOM_SEED = 42


def read_split(name: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with (DATA / name).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("input_text"), str) or not isinstance(row.get("output_text"), str):
                raise ValueError(f"Invalid row {line_number} in {name}")
            texts.append(row["input_text"])
            labels.append(row["output_text"])
    return texts, labels


def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True,
        )),
        ("classifier", classifier),
    ])


def main() -> None:
    X_train, y_train = read_split("train.jsonl")
    X_test_raw, y_test_raw = read_split("test.jsonl")
    train_texts = {text.strip().casefold() for text in X_train}
    kept_test = [(text, label) for text, label in zip(X_test_raw, y_test_raw)
                 if text.strip().casefold() not in train_texts]
    duplicate_rows_excluded = len(X_test_raw) - len(kept_test)
    X_test, y_test = [row[0] for row in kept_test], [row[1] for row in kept_test]
    audit_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True,
    )
    train_audit_vectors = audit_vectorizer.fit_transform(X_train)
    test_audit_vectors = audit_vectorizer.transform(X_test)
    nearest_similarity = cosine_similarity(test_audit_vectors, train_audit_vectors).max(axis=1)
    OUT.mkdir(exist_ok=True)
    labels = sorted(set(y_train) | set(y_test))
    models = {"majority_baseline": make_pipeline(DummyClassifier(strategy="most_frequent"))}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    tuned = {
        "multinomial_nb": (
            make_pipeline(MultinomialNB()),
            {"classifier__alpha": [0.1, 0.5, 1.0, 2.0]},
        ),
        "linear_svc": (
            make_pipeline(LinearSVC(random_state=RANDOM_SEED, max_iter=5000)),
            {"classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        ),
    }
    cv_scores = {}
    for name, (pipeline, params) in tuned.items():
        search = GridSearchCV(pipeline, params, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
        cv_scores[name] = {"best_params": search.best_params_, "cv_macro_f1": search.best_score_}
    summary = []
    with (OUT / "per_class_report.csv").open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=["model", "label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            row = {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
            }
            if name in cv_scores:
                row.update(cv_scores[name])
            if hasattr(model, "decision_function"):
                rank_scores = model.decision_function(X_test)
            else:
                rank_scores = model.predict_proba(X_test)
            row["top3_accuracy"] = top_k_accuracy_score(y_test, rank_scores, k=3, labels=model.classes_)
            summary.append(row)
            details = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
            for label in labels:
                item = details[label]
                writer.writerow({
                    "model": name, "label": label, "precision": item["precision"],
                    "recall": item["recall"], "f1": item["f1-score"], "support": int(item["support"]),
                })
            matrix = confusion_matrix(y_test, predictions, labels=labels)
            with (OUT / f"confusion_{name}.csv").open("w", newline="", encoding="utf-8") as matrix_file:
                csv_writer = csv.writer(matrix_file)
                csv_writer.writerow(["true\\predicted", *labels])
                for label, values in zip(labels, matrix):
                    csv_writer.writerow([label, *values])

    (OUT / "summary.json").write_text(json.dumps({
        "dataset": "gretelai/symptom_to_diagnosis",
        "train_records": len(X_train), "test_records_evaluated": len(X_test),
        "test_records_in_supplied_split": len(X_test_raw),
        "exact_duplicate_test_rows_excluded": duplicate_rows_excluded,
        "remaining_test_rows_nearest_train_cosine_over_090": int((nearest_similarity > 0.90).sum()),
        "remaining_test_rows_nearest_train_cosine_over_080": int((nearest_similarity > 0.80).sum()),
        "labels": len(labels), "random_seed": RANDOM_SEED,
        "results": summary,
        "note": "Closed-set text benchmark only. Scores do not establish clinical validity or diagnostic probability.",
    }, indent=2), encoding="utf-8")
    chosen = models["linear_svc"].named_steps["classifier"].C
    (DATA / "model_config.json").write_text(json.dumps({"linear_svc_C": chosen}, indent=2), encoding="utf-8")
    print((OUT / "summary.json").read_text(encoding="utf-8"))
    print(f"Wrote benchmark artifacts to {OUT}")


if __name__ == "__main__":
    main()
