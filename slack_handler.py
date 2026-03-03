import asyncio
import os

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from agent.steve_jobs import run_po_agent

# ── Slack Bolt App ─────────────────────────────────────────────

bolt_app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)

handler = AsyncSlackRequestHandler(bolt_app)


# ── Block Kit formatter ────────────────────────────────────────

def _bullet_list(items: list[str]) -> str:
    if not items:
        return "_없음_"
    return "\n".join(f"• {item}" for item in items)


def build_result_blocks(idea: str, result: dict) -> list:
    vision = result.get("vision_statement", "")
    gut_reaction = result.get("jobs_gut_reaction", "")
    user_stories = result.get("user_stories", [])
    priorities = result.get("priorities", {})

    stories_text = "\n".join(
        f"• {s.get('full_story', '')}" for s in user_stories
    ) or "_유저 스토리 없음_"

    must = _bullet_list(priorities.get("must_have", []))
    should = _bullet_list(priorities.get("should_have", []))
    could = _bullet_list(priorities.get("could_have", []))
    wont = _bullet_list(priorities.get("wont_have", []))

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📋 Steve Jobs의 PO 분석"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*아이디어:* {idea}\n\n*비전:* _{vision}_",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📖 유저 스토리*\n{stories_text}"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*✅ Must Have*\n{must}"},
                {"type": "mrkdwn", "text": f"*🔵 Should Have*\n{should}"},
            ],
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*💭 Could Have*\n{could}"},
                {"type": "mrkdwn", "text": f"*❌ Won't Have*\n{wont}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*💬 Steve Jobs 한마디*\n_{gut_reaction}_",
            },
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "— Steve Jobs PO Agent"}],
        },
    ]


# ── /steve-jobs slash command ──────────────────────────────────

@bolt_app.command("/steve-jobs")
async def handle_steve_jobs(ack, respond, command):
    # Acknowledge immediately — Slack requires a response within 3 seconds
    await ack()

    idea = command.get("text", "").strip()
    if not idea:
        await respond("사용법: `/steve-jobs [아이디어를 입력하세요]`")
        return

    await respond("Steve Jobs가 검토 중입니다... 🤔")

    async def run_and_respond():
        try:
            result = await run_po_agent(idea)
            blocks = build_result_blocks(idea, result)
            await respond(blocks=blocks, text="Steve Jobs의 PO 분석 완료")
        except Exception as e:
            await respond(f"❌ Steve Jobs가 분석 중 오류가 발생했습니다: {e}")

    asyncio.create_task(run_and_respond())
