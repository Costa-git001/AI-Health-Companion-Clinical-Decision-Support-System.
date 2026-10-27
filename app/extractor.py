"""Transparent phrase-matching baseline for symptom mention extraction.

This is a research prototype, not a diagnostic or clinical model. All matches
are returned for user review, and the exact source phrase is included.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Concept:
    canonical: str
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class Match:
    concept: str
    mention: str
    start: int
    end: int
    status: str
    method: str = "phrase_match_baseline"


# Keep this small vocabulary explicit and auditable until the dataset and
# annotation scheme are selected. It is not a complete clinical terminology.
CONCEPTS = (
    Concept("headache", ("headache", "head pain", "migraine")),
    Concept("fever", ("fever", "high temperature")),
    Concept("cough", ("cough", "coughing")),
    Concept("sore throat", ("sore throat", "throat pain")),
    Concept("shortness of breath", ("shortness of breath", "breathlessness", "difficulty breathing")),
    Concept("chest pain", ("chest pain", "pain in my chest")),
    Concept("nausea", ("nausea", "feeling sick")),
    Concept("vomiting", ("vomiting", "threw up", "throwing up")),
    Concept("diarrhea", ("diarrhea", "loose stools")),
    Concept("abdominal pain", ("abdominal pain", "stomach ache", "stomach pain")),
    Concept("fatigue", ("fatigue", "tiredness", "feeling tired")),
    Concept("dizziness", ("dizziness", "dizzy")),
    Concept("rash", ("rash", "skin rash")),
    Concept("joint pain", ("joint pain", "aching joints")),
    Concept("muscle pain", ("muscle pain", "body aches", "muscle aches")),
    Concept("runny nose", ("runny nose", "nasal discharge")),
    Concept("congestion", ("congestion", "blocked nose")),
)

NEGATION = re.compile(
    r"\b(?:no|not|without|denies?|never|don't have|do not have|doesn't have|does not have)\b[^.!?;]{0,32}$",
    re.IGNORECASE,
)


def _pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


def extract_symptoms(text: str) -> list[dict[str, object]]:
    """Find known symptom phrases and mark simple nearby negation."""
    found: list[Match] = []
    for concept in CONCEPTS:
        for phrase in concept.phrases:
            for match in _pattern(phrase).finditer(text):
                context = text[max(0, match.start() - 40):match.start()]
                status = "negated" if NEGATION.search(context) else "candidate"
                found.append(Match(concept.canonical, match.group(0), match.start(), match.end(), status))

    # Keep one mention per concept (earliest positive mention takes precedence).
    found.sort(key=lambda item: (item.start, item.end - item.start))
    selected: dict[str, Match] = {}
    for item in found:
        existing = selected.get(item.concept)
        if existing is None or (existing.status == "negated" and item.status == "candidate"):
            selected[item.concept] = item
    return [asdict(item) for item in sorted(selected.values(), key=lambda item: item.start)]
