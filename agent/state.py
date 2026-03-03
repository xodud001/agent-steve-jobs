from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class POAgentState(TypedDict):
    """LangGraph state for the Steve Jobs PO Agent workflow."""

    # User input
    user_idea: str

    # Conversation messages (supports add_messages reducer)
    messages: Annotated[list, add_messages]

    # Intermediate outputs
    user_stories: list[dict[str, Any]]
    requirements: dict[str, list[str]]
    priorities: dict[str, list[str]]

    # Steve Jobs specific
    vision_statement: str        # The "why" behind the product
    jobs_initial_reaction: str   # Jobs' gut reaction to the idea
    simplicity_cuts: list[str]   # Things Jobs would ruthlessly cut

    # Final structured output
    result: dict[str, Any]

    # Workflow control
    current_step: str
    error: str | None
