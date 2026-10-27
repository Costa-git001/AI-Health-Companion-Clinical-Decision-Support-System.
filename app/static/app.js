const description = document.querySelector('#description');
const extractButton = document.querySelector('#extract-button');
const errorBox = document.querySelector('#form-error');
const emptyState = document.querySelector('#empty-state');
const reviewArea = document.querySelector('#review-area');
const symptomList = document.querySelector('#symptom-list');
const summaryButton = document.querySelector('#summary-button');
const summaryPanel = document.querySelector('#summary-panel');
const rankButton = document.querySelector('#rank-button');
const rankPanel = document.querySelector('#rank-panel');

let reviewedSymptoms = [];
let currentMatches = [];

description.addEventListener('input', () => {
  document.querySelector('#char-count').textContent = `${description.value.length} / 4000`;
});

extractButton.addEventListener('click', async () => {
  const text = description.value.trim();
  errorBox.hidden = true;
  if (text.length < 3) {
    errorBox.textContent = 'Add a short description before organizing your notes.';
    errorBox.hidden = false;
    return;
  }
  extractButton.disabled = true;
  extractButton.querySelector('span:first-child').textContent = 'Organizing…';
  try {
    const response = await fetch('/api/extract', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    if (!response.ok) throw new Error('The notes could not be processed. Please try again.');
    const data = await response.json();
    currentMatches = data.matches;
    rankButton.disabled = false;
    rankPanel.hidden = true;
    reviewedSymptoms = data.matches.filter((match) => match.status !== 'negated').map((match) => ({
      name: match.concept,
      evidence: `Found “${match.mention}” in your description`,
      isMatch: true
    }));
    drawReview();
    summaryPanel.hidden = true;
  } catch (error) {
    errorBox.textContent = error.message || 'Unable to organize the notes right now.';
    errorBox.hidden = false;
  } finally {
    extractButton.disabled = false;
    extractButton.querySelector('span:first-child').textContent = 'Organize my notes';
  }
});

rankButton.addEventListener('click', async () => {
  const error = document.querySelector('#rank-error');
  error.hidden = true;
  rankButton.disabled = true;
  rankButton.textContent = 'Ranking benchmark labels…';
  try {
    const response = await fetch('/api/rank', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: description.value.trim()})
    });
    if (!response.ok) throw new Error('The ranking model could not process this text.');
    const data = await response.json();
    const results = document.querySelector('#rank-results');
    results.replaceChildren();
    data.candidates.forEach((candidate) => {
      const row = document.createElement('article');
      row.className = 'rank-result';
      const number = document.createElement('span');
      number.className = 'rank-number';
      number.textContent = String(candidate.rank).padStart(2, '0');
      const body = document.createElement('div');
      body.className = 'rank-result-body';
      const label = document.createElement('strong');
      label.textContent = candidate.label;
      const detail = document.createElement('span');
      detail.textContent = candidate.text_signals.length
        ? `Text signals: ${candidate.text_signals.join(', ')}`
        : 'No distinct text signals surfaced for this label.';
      body.append(label, detail);
      row.append(number, body);
      results.append(row);
    });
    rankPanel.hidden = false;
  } catch (cause) {
    error.textContent = cause.message || 'Ranking is temporarily unavailable.';
    error.hidden = false;
    rankPanel.hidden = false;
  } finally {
    rankButton.disabled = false;
    rankButton.innerHTML = 'Explore AI ranked labels <span>→</span>';
  }
});

function drawReview() {
  emptyState.hidden = true;
  reviewArea.hidden = false;
  summaryButton.disabled = false;
  symptomList.replaceChildren();
  if (!reviewedSymptoms.length) {
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = 'No known phrases were found. Add any symptoms or details you want to include.';
    symptomList.append(note);
  }
  reviewedSymptoms.forEach((symptom, index) => {
    const row = document.createElement('div');
    row.className = 'symptom-item';
    const copy = document.createElement('div');
    copy.className = 'symptom-copy';
    const check = document.createElement('span');
    check.className = 'checkmark';
    check.textContent = '✓';
    const text = document.createElement('div');
    text.style.minWidth = '0';
    const name = document.createElement('span');
    name.className = 'symptom-name';
    name.textContent = symptom.name;
    const evidence = document.createElement('span');
    evidence.className = 'symptom-evidence';
    evidence.textContent = symptom.evidence;
    text.append(name, evidence);
    copy.append(check, text);
    const remove = document.createElement('button');
    remove.className = 'remove-button';
    remove.type = 'button';
    remove.setAttribute('aria-label', `Remove ${symptom.name}`);
    remove.textContent = '×';
    remove.addEventListener('click', () => {
      reviewedSymptoms.splice(index, 1);
      drawReview();
    });
    row.append(copy, remove);
    symptomList.append(row);
  });
  const negatives = currentMatches.filter((match) => match.status === 'negated').length;
  document.querySelector('#match-summary').textContent = currentMatches.length
    ? `${currentMatches.length} phrase match${currentMatches.length === 1 ? '' : 'es'} found; ${negatives} marked as negated and excluded. Please verify the list.`
    : 'No known phrase matched. You can add details manually. This baseline may miss words and context.';
}

function addSymptom() {
  const input = document.querySelector('#add-symptom');
  const name = input.value.trim();
  if (!name) return;
  if (!reviewedSymptoms.some((item) => item.name.toLowerCase() === name.toLowerCase())) {
    reviewedSymptoms.push({name, evidence: 'Added by you', isMatch: false});
  }
  input.value = '';
  drawReview();
}

document.querySelector('#add-button').addEventListener('click', addSymptom);
document.querySelector('#add-symptom').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') addSymptom();
});

summaryButton.addEventListener('click', () => {
  const date = new Intl.DateTimeFormat(undefined, {dateStyle: 'long'}).format(new Date());
  document.querySelector('#summary-date').textContent = date;
  document.querySelector('#summary-duration').textContent = document.querySelector('#duration').value.trim() || 'Not provided';
  document.querySelector('#summary-severity').textContent = document.querySelector('#severity').value || 'Not provided';
  document.querySelector('#summary-description').textContent = description.value.trim() || 'No description provided.';
  const list = document.querySelector('#summary-items');
  list.replaceChildren();
  if (!reviewedSymptoms.length) {
    const item = document.createElement('li');
    item.textContent = 'No symptoms listed';
    list.append(item);
  } else {
    reviewedSymptoms.forEach((symptom) => {
      const item = document.createElement('li');
      item.textContent = `${symptom.name} — ${symptom.evidence}`;
      list.append(item);
    });
  }
  summaryPanel.hidden = false;
  summaryPanel.scrollIntoView({behavior: 'smooth', block: 'start'});
});

document.querySelector('#print-button').addEventListener('click', () => window.print());
document.querySelector('#close-summary').addEventListener('click', () => { summaryPanel.hidden = true; });
