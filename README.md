# AI Health Companion

A local research prototype that extracts symptom terms, ranks three labels from a fixed benchmark using a trained text classifier, and prepares a pre-visit summary. Its ranked labels are not diagnoses, clinical probabilities, or treatment advice. The app does not triage, interpret tests, or store patient records.

## Run locally

1. Install Python 3.11 or newer.
2. In this folder, create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   uvicorn app.main:app --reload
   ```

4. Open http://127.0.0.1:8000.

The service processes submitted text in memory. It does not write request bodies or records to a database. The browser keeps the current draft only until the page is refreshed or closed. Use synthetic information for demonstrations; this prototype is not approved for real patient data.

## Prototype workflow

1. Enter a short, de-identified symptom description.
2. Run the phrase-matching baseline and review, remove, or add extracted symptom concepts.
3. Optionally run the AI ranking demo to see three ranked labels from a fixed 22-label benchmark.
4. Add duration and severity as reported by the user.
5. Generate and print or save a pre-visit summary.

Every extracted phrase is a candidate for user review. A matching score is not a diagnosis or probability of disease.

## AI component and dataset

`app/condition_ranker.py` trains a TF-IDF + Linear SVM classifier on the included training split. It ranks labels from the benchmark only and reports no probabilities. Its “text signals” are model feature contributions, not clinical evidence. `app/extractor.py` remains a transparent phrase-matching baseline for symptom normalization.

The included [Gretel symptom-to-diagnosis dataset](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) has 1,065 English descriptions mapped to 22 labels (853 train, 212 test). The dataset card reports Apache 2.0 and says the descriptions were adapted from Kaggle's Symptom2Disease and updated with an LLM. It is small, English-only, and does not establish clinician-confirmed outcome labels or represent a validated patient population. Use it only for a bounded classroom text-classification benchmark, not for real-patient diagnosis or probability claims. See `data/README.md` and `data/LICENSE-2.0.txt`.

Run the reproducible closed-set benchmark:

```powershell
python scripts/benchmark_ranker.py
```

The script tunes Multinomial Naive Bayes and Linear SVM using stratified five-fold cross-validation inside the training split, compares them with a majority-label baseline, then evaluates once against the supplied test split. An exact normalized text duplicate across train and test is excluded from metric calculation and disclosed in the summary; the source files remain unchanged. It writes overall metrics, per-class metrics, and confusion matrices in `results/`. It reports accuracy, macro F1, top-3 accuracy, per-class precision/recall/F1, and confusion matrices. These are benchmark metrics, not clinical validation. Preserve the fixed test set and do not tune against it.

The current benchmark run evaluates 211 test examples after excluding one exact duplicate: Linear SVM accuracy 94.3%, macro F1 94.2%, top-3 accuracy 99.1%; Multinomial Naive Bayes accuracy 93.8%, macro F1 93.7%, top-3 accuracy 98.6%. Nearest-neighbor text audit also flags highly similar train/test pairs, so the high scores may reflect dataset-specific wording patterns. See `results/summary.json`; these figures must not be presented as clinical accuracy.

The `scripts/evaluate_extractor.py` utility separately calculates exact-concept precision, recall and F1 against an approved annotated JSONL symptom-extraction set. No such annotations are bundled.

## API

- `GET /api/health` — service status.
- `POST /api/extract` — accepts `{"text":"..."}`; returns extracted candidate concepts and exact evidence spans.
- `POST /api/rank` — accepts `{"text":"..."}`; returns three ordered labels from the fixed dataset benchmark and text features that influenced each ranking.

The web client creates the summary locally from reviewed values; summary content is not submitted to the service.

## Current scope and limitations

- No urgency classifier, medication or lab interpretation, image processing, authentication, cloud service, or clinician account.
- No persistent patient profile or cross-device sharing. The user can print the reviewed summary or save it as PDF through the browser.
- The phrase list is a small prototype vocabulary and will miss paraphrases, spelling variants, context, and concepts outside that list.
- Negation handling is deliberately limited; always review and correct the candidates.
- The app is a student research prototype and has not been clinically validated.

## Suggested next research decisions

1. Confirm with the supervisor that the dataset benchmark is acceptable for a research demo and agree how to describe its limits.
2. Establish ethics and data-protection requirements before collecting information from people.
3. Seek a better documented, clinically grounded and representative dataset or an approved clinical advisor before making clinical claims.
4. Validate usability with representative users only after required institutional approval.

## Structure

- `app/main.py` — FastAPI app and API boundary.
- `app/extractor.py` — transparent NLP baseline.
- `app/condition_ranker.py` — TF-IDF + Linear SVM closed-label ranking model.
- `data/` — licensed train/test data split, attribution and provenance notes.
- `app/static/` — patient-facing prototype.
- `scripts/benchmark_ranker.py` — reproducible classifier comparison and metrics.
- `scripts/evaluate_extractor.py` — reproducible metric calculation for approved annotations.
- `docs/ARCHITECTURE.md` — architecture and data flow.
