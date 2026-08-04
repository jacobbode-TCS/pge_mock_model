from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from llm.llm import llm

# For testing, we can allow all prompts to be routed to the orchestrator
ALLOW_ALL_PROMPTS = False

class InputEvaluator(BaseModel):
    next_agent: Literal["orchestrator", "end"]

if ALLOW_ALL_PROMPTS:
    SYSTEM_PROMPT = """
        You are an input evaluation agent in a multi-agent assistant chat. Route the user's request to the orchestrator.
        """
else:
    SYSTEM_PROMPT = """
        You are an input evaluation agent in a multi-agent assistant chat.

        Determine whether the user's request is relevant to the following topics.
        If it is relevant, route it to the orchestrator.
        If it is not relevant, route it to the end.

        Topics:
        - bird image classification or bird recognition
        - construction guidance or assembly questions for utility products or services.
        - cost estimation, budgeting, or validation for utility products or services.
        - general knowledge questions about utility products or services.

        Return only one of these values:
        - orchestrator
        - end
        """

router = llm.with_structured_output(InputEvaluator)


def decide_if_continue(request: str,) -> InputEvaluator:
    """Decide whether to continue the workflow for the given request."""


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{request}"),
        ]
    )
    response = router.invoke(prompt.format_messages(request=request))
    return response