import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "local-model")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model=OPENAI_MODEL,
)

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
- Create new artifacts, plans, workflows, designs, documents, code.

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

def decide_next_agent(
    request: str,
    llm_client=None,
) -> Orchestrator:

    client = llm_client or router

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{request}"),
        ]
    )

    return client.invoke(
        prompt.format_messages(request=request)
    )