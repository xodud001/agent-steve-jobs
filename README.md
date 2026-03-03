# AI PO Agent

LangGraph + Anthropic(claude-opus-4-5) 기반 AI Product Owner 에이전트.
아이디어를 입력하면 유저 스토리 → 요구사항 → MoSCoW 우선순위를 자동으로 생성합니다.

## 아키텍처

```
사용자 입력
    │
    ▼
analyze_idea ──► tools_analyze (write_user_stories)
    │
    ▼
process_stories ──► tools_stories (define_requirements)
    │
    ▼
process_requirements ──► tools_requirements (prioritize_moscow)
    │
    ▼
compile_result ──► JSON 반환
```

## 빠른 시작

### 1. 의존성 설치

```bash
cd po-agent
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에서 ANTHROPIC_API_KEY 값을 실제 키로 교체
```

### 3. 서버 실행

```bash
python main.py
# 또는
uvicorn main:app --reload
```

서버가 `http://localhost:8000`에서 시작됩니다.

---

## API

### `GET /health`

서버 상태 확인.

```bash
curl http://localhost:8000/health
```

응답:
```json
{
  "status": "ok",
  "model": "claude-opus-4-5",
  "api_key_configured": true
}
```

---

### `POST /run` — 동기 응답

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"idea": "온라인 쇼핑몰 장바구니 기능"}'
```

응답:
```json
{
  "status": "success",
  "result": {
    "summary": "...",
    "user_stories": [...],
    "requirements": {
      "functional": [...],
      "non_functional": [...]
    },
    "priorities": {
      "must_have": [...],
      "should_have": [...],
      "could_have": [...],
      "wont_have": [...]
    }
  }
}
```

---

### `POST /run` — SSE 스트리밍

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"idea": "온라인 쇼핑몰 장바구니 기능", "stream": true}' \
  --no-buffer
```

스트림 이벤트:
```
data: {"step": "analyze_idea", "message": "아이디어 분석 중..."}

data: {"step": "tools_analyze", "message": "유저 스토리 도구 실행 중..."}

...

data: {"step": "done", "result": {...}}

data: [DONE]
```

---

### Swagger UI

서버 실행 후 브라우저에서 확인:
`http://localhost:8000/docs`

---

## 프로젝트 구조

```
po-agent/
├── main.py              # FastAPI 서버 (엔드포인트 정의)
├── agent/
│   ├── __init__.py
│   ├── steve_jobs.py      # LangGraph 그래프 + 실행 함수
│   ├── tools.py         # LangChain 도구 (유저 스토리/요구사항/우선순위)
│   └── state.py         # POAgentState TypedDict
├── .env.example
├── requirements.txt
└── README.md
```
