# Signal Detector

Pharmacovigilance dashboard (no Streamlit): FastAPI backend + static HTML/JS frontend.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

- Start API (port 8000):

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Serve frontend (port 8080):

```bash
python -m http.server 8080 --directory public
```

Visit http://localhost:8080 in your browser.

## Endpoints

- GET `/api/kpis`
- GET `/api/events`
- GET `/api/heatmap`
- GET `/api/signals`
- GET `/api/alerts`
- GET `/api/nlp`

## Data

Initial synthetic dataset in `data/synthetic_reports.csv`. Backend expands with additional synthetic rows on first run.
