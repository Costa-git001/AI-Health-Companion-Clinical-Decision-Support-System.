from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.extractor import extract_symptoms
from app.condition_ranker import get_ranker


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="AI Health Companion Research Prototype",
    description="Local research prototype for symptom extraction, fixed-benchmark text ranking, and pre-visit summary preparation.",
    version="0.1.0",
)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class RankRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-research-prototype"}


@app.post("/api/extract")
def extract(request: ExtractRequest) -> dict[str, object]:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Enter a short symptom description.")
    return {
        "matches": extract_symptoms(request.text),
        "notice": "These phrase matches are candidates for review, not a diagnosis or medical advice.",
    }


@app.post("/api/rank")
def rank(request: RankRequest) -> dict[str, object]:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Enter a short symptom description.")
    results = get_ranker().rank(request.text.strip(), limit=3)
    return {
        "candidates": results,
        "dataset": {
            "name": "Gretel symptom_to_diagnosis",
            "train_records": 853,
            "label_count": 22,
            "language": "English",
        },
        "notice": "Closed-set benchmark labels only. These rankings are not diagnoses, clinical probabilities, or medical advice.",
    }


app.mount("/assets", StaticFiles(directory=BASE_DIR / "static"), name="assets")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")
