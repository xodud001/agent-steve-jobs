import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from agent.steve_jobs import run_po_agent, stream_po_agent  # noqa: E402
from slack_handler import handler as slack_handler  # noqa: E402


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    for key, label in [
        ("ANTHROPIC_API_KEY", "Anthropic API key"),
        ("SLACK_BOT_TOKEN", "Slack bot token"),
        ("SLACK_SIGNING_SECRET", "Slack signing secret"),
    ]:
        if not os.environ.get(key):
            raise RuntimeError(f"{label} is not set. Check your .env file.")
    print("Steve Jobs Agent server starting")
    yield
    print("Steve Jobs Agent server stopped")


app = FastAPI(
    title="Steve Jobs AI Agent",
    description="LangGraph-based AI agent that thinks and acts like Steve Jobs — challenging vision, writing user stories, defining requirements, and prioritizing with ruthless focus.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schema ─────────────────────────────────────────────────────

class RunRequest(BaseModel):
    idea: str
    stream: bool = False

    model_config = {"json_schema_extra": {"example": {"idea": "A music player that changes how people experience their entire music library", "stream": False}}}


# ── REST endpoints ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Server health check"""
    return {
        "status": "ok",
        "agent": "Steve Jobs",
        "model": "claude-sonnet-4-6",
        "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "slack_configured": bool(os.environ.get("SLACK_BOT_TOKEN")),
    }


@app.post("/run")
async def run_agent(request: RunRequest):
    """
    Run the Steve Jobs agent on your idea.

    - **idea**: The product idea or feature request to analyze
    - **stream**: True for SSE streaming, False for synchronous JSON response
    """
    if not request.idea.strip():
        raise HTTPException(status_code=400, detail="idea field is empty.")

    if request.stream:
        return StreamingResponse(
            stream_po_agent(request.idea),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await run_po_agent(request.idea)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Slack endpoints ────────────────────────────────────────────

@app.post("/slack/events")
async def slack_events(req: Request):
    """Entry point for all Slack events and slash commands."""
    return await slack_handler.handle(req)


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
