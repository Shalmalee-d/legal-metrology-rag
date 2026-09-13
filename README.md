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
- FastAPI
- APScheduler
- HTTPX
- BeautifulSoup
- PyMuPDF
- Pydantic
- PostgreSQL