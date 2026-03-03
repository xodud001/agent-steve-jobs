import json
import logging
import re
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def challenge_vision(idea: str) -> str:
    """
    Steve Jobs challenges the fundamental premise of the idea.
    Forces clarity on the WHY before diving into features.

    Args:
        idea: The raw product idea or feature request

    Returns:
        Vision challenge framework from Jobs' perspective
    """
    return f"""Idea under scrutiny: {idea}

Steve, before a single line of code is written, ask:

1. IS THIS TRULY REVOLUTIONARY?
   - Does it change how people live, work, or create?
   - Or is it just another incremental feature nobody asked for?
   - "We don't ship crap." — What would make this insanely great?

2. WHAT'S THE EMOTIONAL CORE?
   - In one sentence: what feeling does this give the user?
   - Not "it lets users do X" — but "it makes users feel Y"
   - "The most powerful person in the world is the storyteller."

3. WHAT SHOULD WE *NOT* BUILD?
   - List the obvious features that dilute the core experience
   - Simplicity is the ultimate sophistication
   - "Deciding what not to do is as important as deciding what to do."

4. STATE THE VISION:
   - Write a single, unforgettable vision statement (≤ 2 sentences)
   - It should make engineers excited and competitors nervous
   - This is the north star. Everything else is noise."""


@tool
def write_user_stories(feature_description: str) -> str:
    """
    Write user stories the Steve Jobs way — centered on human desire,
    not feature checklists.

    Args:
        feature_description: The product feature or idea

    Returns:
        Jobs-style user story writing framework
    """
    return f"""Feature: {feature_description}

Steve's rules for user stories:

"You've got to start with the customer experience and work backward
to the technology — not the other way around."

Write 3–5 user stories that:
- Speak to DESIRE, not just functionality
  Bad:  "As a user, I want to upload a file."
  Good: "As a creative professional, I want my work to be in the cloud
         the moment I save it, so I never think about syncing again."

- Each story must pass the "So What?" test
  Ask "so what?" after every benefit. If the answer is shallow, dig deeper.

- Capture the MAGIC MOMENT
  What's the instant the user thinks "this just works"?

- Format:
  As a [specific person, not generic "user"],
  I want to [visceral action],
  so that [meaningful life impact]."""


@tool
def define_requirements(user_stories: str) -> str:
    """
    Define requirements with Jobs' obsession for simplicity and elegance.
    Every requirement must earn its place.

    Args:
        user_stories: The user stories already written

    Returns:
        Jobs-style requirements definition framework
    """
    return f"""User stories in context:
{user_stories}

Steve's requirements philosophy:

"Simple can be harder than complex. You have to work hard to get
your thinking clean to make it simple."

FUNCTIONAL REQUIREMENTS — The Soul of the Product:
- Each requirement must directly serve a user story
- If you can't explain it to a 10-year-old, simplify it
- Write it as a capability, not an implementation detail
  Bad:  "The system shall implement a REST API for data retrieval."
  Good: "Users get their data instantly, on any device, without waiting."

NON-FUNCTIONAL REQUIREMENTS — The Invisible Excellence:
- Performance: "It should feel instantaneous." (< 100ms perceived latency)
- Reliability: "It just works." — always, without babysitting
- Design: "It must be so intuitive that there's no manual."
- Security: Protection should be invisible to the user

JOBS' RAZOR — Apply to every requirement:
"Does removing this make the product worse for the user?
If not, cut it."

List only what survives the razor."""


@tool
def prioritize_moscow(requirements: str) -> str:
    """
    Apply MoSCoW prioritization with Steve Jobs' ruthless focus.
    Must Haves are the product's soul — everything else is negotiable.

    Args:
        requirements: The defined requirements

    Returns:
        Jobs-style MoSCoW prioritization framework
    """
    return f"""Requirements to prioritize:
{requirements}

Steve's prioritization doctrine:

"Real artists ship. But they ship the RIGHT thing."

MUST HAVE — "The Soul" (≤ 3 items):
- If this isn't in v1, the product doesn't exist
- This is what you demo on stage. This is what makes people gasp.
- Be brutally honest: most things you think are "must have" are not.
- Ask: "Would a user return the product if this was missing?"

SHOULD HAVE — "The Polish" (≤ 3 items):
- Makes the experience delightful, not just functional
- Important, but the core vision survives without it
- These ship in the next release if they don't make the deadline

COULD HAVE — "The Extras" (minimize):
- Nice ideas that don't move the needle
- The graveyard of good-but-not-great features
- If you're debating whether it belongs here, it probably belongs in Won't Have

WON'T HAVE — "The Cuts" (be aggressive):
- Everything that dilutes focus
- "Deciding what not to do is as important as deciding what to do."
- No apologies. Great products say no to almost everything.

For each item, state WHY it belongs in that category."""


def parse_json_from_response(text: str) -> dict:
    """Extract JSON block from LLM response."""
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, text)

    logger.info("[parse_json] found %d code block(s)", len(matches))

    if matches:
        try:
            result = json.loads(matches[0])
            logger.info("[parse_json] code block parsed OK, keys: %s", list(result.keys()))
            return result
        except json.JSONDecodeError as e:
            logger.warning("[parse_json] code block JSON decode failed: %s", e)
            logger.warning("[parse_json] block content (first 500):\n%s", matches[0][:500])

    try:
        result = json.loads(text)
        logger.info("[parse_json] bare JSON parsed OK")
        return result
    except json.JSONDecodeError as e:
        logger.warning("[parse_json] bare JSON also failed: %s", e)
        return {}


ALL_TOOLS = [challenge_vision, write_user_stories, define_requirements, prioritize_moscow]
