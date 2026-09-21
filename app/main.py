from fastapi import FastAPI

from app.api.routes import router
from app.scheduler.scheduler import start_scheduler


app = FastAPI(title="Legal Metrology Regulatory RAG")
app.include_router(router)


@app.on_event("startup")
def start_background_regulatory_updates() -> None:
    """Start weekly regulatory monitoring independently of product scans."""
    start_scheduler()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
