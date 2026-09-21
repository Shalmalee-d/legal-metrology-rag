# Legal Metrology RAG

Automated Regulatory RAG pipeline for the Legal Metrology Packaged Commodities Rules, 2011.

## Purpose

The system periodically checks official government sources for new or updated Legal Metrology regulations, extracts relevant regulatory information, converts it into structured compliance rules, validates those rules, and provides them to the Rule Engine.

## Architecture

Official Government Sources
        ↓
Scheduled Fetch
        ↓
Change Detection
        ↓
Document Extraction
        ↓
Regulatory Rule Extraction
        ↓
Rule Validation
        ↓
Versioned Rule Repository
        ↓
Rule Engine

## Tech Stack

- Python
- APScheduler
- HTTPX
- BeautifulSoup
- PyMuPDF
- Pydantic
- JSON rule repository (replaceable with a database adapter later)

## Regulatory update flow

```text
Official Department of Consumer Affairs sources
  -> scheduled fetch (APScheduler)
  -> SHA-256 change detection (NEW / CHANGED / UNCHANGED)
  -> PyMuPDF extraction, with per-page Tesseract OCR fallback
  -> careful text cleaning and page-aware chunks
  -> deterministic, evidence-backed rule extraction
  -> validation and evidence/page merging
  -> versioned JSON rule repository
  -> separate Rule Engine
```

The RAG service owns regulatory knowledge updates. It supplies structured
`ComplianceRule` records to the separate product-scanning Rule Engine.

## OCR and bilingual documents

Normal PDF text is extracted directly with PyMuPDF. Sparse or image-only pages
fall back to Tesseract OCR and are labelled with `method: "ocr"`; good direct
text is never replaced. OCR defaults to `eng+hin`. If Hindi traineddata is not
installed, the extractor continues with English when available.

Install Tesseract and put it on `PATH`. On Windows, set `TESSERACT_CMD` only if
it is not on PATH. Install `hin.traineddata` in Tesseract's `tessdata` folder
to enable Hindi OCR. Useful configuration variables are:

```text
TESSERACT_CMD=/path/to/tesseract
TESSERACT_LANG=eng+hin
OCR_RENDER_SCALE=3
OCR_TIMEOUT_SECONDS=120
RAG_CHECK_INTERVAL_DAYS=7
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
RAG_CHUNK_SIZE=2400
```

## Running

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
pytest
python -m app.main
```

`app.scheduler.scheduler.start_scheduler()` performs one initial lightweight
check and starts the independent weekly scheduler. Both the scheduler and
`POST /api/rag/refresh` call the same `run_rag_update()` function. Known
documents are checked using stable HTTP metadata before a PDF download;
SHA-256 confirms any candidate change. OCR and local Qwen extraction run only
for new, changed, or bootstrap documents.

## Local Qwen and API

Install and start Ollama, then run these exact local commands:

```powershell
ollama list
ollama pull qwen3:4b
uvicorn app.main:app --reload
.\venv\Scripts\python.exe -m pytest -q
```

The service calls Ollama's local HTTP API only; no cloud API key is used. Qwen
proposes source-backed JSON rules, while validation remains responsible for
activation and product compliance decisions.
The public backend integration is `OllamaRuleGenerator` from
`app.extraction.llm_rule_generator`; its `generate_rules()` method uses the
configured local endpoint and model.

The scheduler runs one lightweight check at startup and then every
`RAG_CHECK_INTERVAL_DAYS=7` days. Normal refreshes invoke Qwen only for a NEW
or CHANGED PDF after metadata and SHA-256 checks; it is never invoked for
unchanged checks or product scanning. A controlled `rebuild=True` processes
every discovered scoped document and atomically replaces the repository only
after the complete non-empty validated candidate succeeds. The frontend can call:

- `POST /api/rag/refresh` to queue a non-blocking check, then poll status.
- `GET /api/rag/status` for update status.
- `GET /api/rules/active` for active validated rules.

Every repository write re-runs deterministic validation; invalid rules cannot
be persisted or returned as active. Incremental updates retire rules absent
from a changed source document while retaining unrelated source history.

Failed downloads, OCR, Ollama, validation, empty extraction, or repository updates keep the last
valid active-rule snapshot. Rule versions are retained and older versions are
marked superseded.

## Dynamic source updates

Every refresh discovers the currently listed official documents; it does not
depend on a fixed document count. Discovery uses both link text and nearby page
category context, and admits only Legal Metrology (Packaged Commodities)
materials while explicitly excluding General Rules, National Standards,
Approval of Models, and Numeration categories.

`data/registry/document_registry.json` is a local, Git-ignored source registry.
It records stable URL-derived document identities, category, fingerprints,
checks, processing state, and a `no_longer_listed` source-tracking state. New
and changed PDFs alone enter OCR, local Qwen, and validation; unchanged records
return on the fast path. Original Hindi (or other language) evidence, source,
and page provenance remain on accepted rules.

For a deliberate clean rebuild after reviewing the source set, run the local
application environment and invoke `run_rag_update(rebuild=True)`. A rebuild
only replaces the existing JSON snapshot after the complete discovered scoped
set succeeds; a failed or partial rebuild retains the prior repository.

### Change-detection safety

Routine refreshes use a fast HTTP signature path when a strong ETag is available,
or when Last-Modified and Content-Length both match the processed version. If
those signals are weak or absent, the PDF is downloaded and SHA-256 is used to
confirm whether the content actually changed. A failed source HEAD check is not
silently treated as unchanged.

A controlled rebuild also refuses to replace the existing repository with an
empty rule snapshot. Existing rules remain intact unless the complete rebuild
produces a non-empty validated rule set.
