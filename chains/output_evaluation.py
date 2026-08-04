from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from llm.llm import llm


class OutputEvaluator(BaseModel):
    next_agent: Literal["review", "end"]

ALWAYS_REVIEW = False

if ALWAYS_REVIEW:
    SYSTEM_PROMPT = "No matter what you are given, always just return the value 'review', this is for testing purposes."
else:
    SYSTEM_PROMPT = """
    You are an output evaluation agent in a multi-agent assistant chat.

    Judge the latest assistant output against the user's request.
    If the output is relevant, complete, and directly answers the request, route it to the end.
    If it is incomplete, irrelevant, inaccurate, or poorly formed, route it to the review.

    Return only one of these values:
    - end
    - review
    """



router = llm.with_structured_output(OutputEvaluator)


def decide_if_continue(request: str, output: str | None = None) -> OutputEvaluator:
    """Decide whether the latest output is good enough to finish the workflow."""

    evaluation_text = f"User request:\n{request}\n\nLatest output:\n{output or ''}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{request}"),
        ]
    )
    response = router.invoke(prompt.format_messages(request=evaluation_text))
    return response