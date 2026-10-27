# Dataset notes

## Gretel symptom to diagnosis

- Source: [gretelai/symptom_to_diagnosis on Hugging Face](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)
- License identified by the dataset card: Apache 2.0; a copy is included as `LICENSE-2.0.txt`.
- Contents documented by the card: 1,065 English text descriptions, 22 single diagnosis labels, fixed 853-record training and 212-record test splits.
- Provenance: adapted from Kaggle's Symptom2Disease; the card says the text was filtered and updated with an LLM to make it more patient-like.
- Research use here: closed-label educational text-classification benchmark only.

## Limitations

The card does not establish that each label is a clinician-confirmed outcome or give a sampling frame representing people seeking care. LLM modification can introduce artifacts. There are only about 40 training and 7–10 test examples per label; the task is English-only and every example has exactly one label. It does not represent prevalence, co-occurring conditions, missing information, age-specific risks, or uncertainty in real consultations. It cannot support claims that the app diagnoses patients, gives calibrated probability estimates, or is suitable for clinical decisions.

Keep the supplied train/test files separate. Use cross-validation only inside the training split to select hyperparameters; report the fixed test split once. Record these provenance limits alongside any metric.

An exact normalized text duplicate occurs across the supplied training and test splits. The benchmark script excludes the duplicate test row from metric calculations and records this exclusion in `results/summary.json`; the original dataset files are kept unchanged. Near-duplicate wording may remain and should be discussed as a limitation.
