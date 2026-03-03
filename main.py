import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from agent.steve_jobs import run_po_agent, stream_po_agent  # noqa: E402


# ── 앱 수명주기 ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
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


# ── 스키마 ─────────────────────────────────────────────────────

class RunRequest(BaseModel):
    idea: str
    stream: bool = False

    model_config = {"json_schema_extra": {"example": {"idea": "A music player that changes how people experience their entire music library", "stream": False}}}


# ── 엔드포인트 ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "ok",
        "agent": "Steve Jobs",
        "model": "claude-sonnet-4-6",
        "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
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


# ── 진입점 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
