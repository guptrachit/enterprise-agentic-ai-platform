from fastapi import FastAPI

app = FastAPI(
    title="Enterprise Agentic AI Platform",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "application": "Enterprise Agentic AI Platform",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
