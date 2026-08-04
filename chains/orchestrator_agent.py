from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from llm.llm import llm

class Orchestrator(BaseModel):
    next_agent: Literal[
        "image_analysis",
        "construction",
        "estimation",
        "knowledge_search",
        "review",
    ]
SYSTEM_PROMPT = """
You are a routing agent in a multi-agent system.

Select the single best agent for handling the request.

Agents:

image_analysis
- Analyze images, photos, screenshots, diagrams.
- Identify objects, defects, species, visual features.

construction
- Answers questions related to the usage and assembly of equipment.

estimation
- Cost estimation, pricing, validation, verification,
  double-checking calculations, budgeting.

knowledge_search
- Answers informational questions, 
- Uses internal model knowledge when sufficient. 
- Uses external search when current, specialized, or uncertain information is required.

review
- Use when none clearly apply.

Return only the selected agent.
"""

router = llm.with_structured_output(Orchestrator)

def decide_next_agent(request: str, conversation=None) -> Orchestrator:
    """Decide the next agent to route to.

    If `conversation` is provided the system+user messages are appended to
    the shared conversation history and the router is invoked with the
    accumulated messages so the model can consider prior turns.
    """

    client = router

    # If a conversation is provided, add the routing system prompt and the
    # user request to the shared history and invoke using that history.
    if conversation is not None:
        conversation.add_system(SYSTEM_PROMPT)
        conversation.add_user(request)
        return client.invoke(conversation.history)

    # Fallback to the single-shot prompt behaviour
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{request}"),
        ]
    )

    return client.invoke(prompt.format_messages(request=request))