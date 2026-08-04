from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from llm.llm import llm


class OutputEvaluator(BaseModel):
    next_agent: Literal["review", "end"]


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


def _coerce_output_evaluator(response) -> OutputEvaluator:
    if isinstance(response, OutputEvaluator):
        return response

    if hasattr(response, "parsed") and response.parsed is not None:
        return response.parsed

    if hasattr(response, "content"):
        content = response.content
    elif isinstance(response, dict):
        content = response.get("next_agent") or response.get("content")
    else:
        content = str(response)

    if isinstance(content, str):
        normalized = content.strip().lower()
        if normalized in {"end", "review"}:
            return OutputEvaluator(next_agent=normalized)

    raise ValueError(f"Unable to parse output evaluation response: {response}")


def decide_if_continue(request: str, conversation=None, llm_client=None, output: str | None = None) -> OutputEvaluator:
    """Decide whether the latest output is good enough to finish the workflow."""

    if llm_client is None:
        client = router
    else:
        client = llm_client

    evaluation_text = f"User request:\n{request}\n\nLatest output:\n{output or ''}"

    if conversation is not None:
        conversation.add_system(SYSTEM_PROMPT)
        conversation.add_user(evaluation_text)
        response = client.invoke(conversation.history)
        return _coerce_output_evaluator(response)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{request}"),
        ]
    )
    response = client.invoke(prompt.format_messages(request=evaluation_text))
    return _coerce_output_evaluator(response)