import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "local-model")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model=OPENAI_MODEL,
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None


def estimate_cost(request: str, max_results: int = 5) -> dict:
    """Estimate a ballpark cost for a user request by searching the web and summarizing findings."""
    if not TAVILY_API_KEY or tavily_client is None:
        return {
            "request": request,
            "estimate": "Tavily API key is not configured.",
            "sources": [],
        }

    search_results = tavily_client.search(
        query=f"ballpark cost estimate for {request}",
        search_depth="basic",
        max_results=max_results,
    )

    sources = []
    for item in search_results.get("results", []):
        sources.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You provide rough cost estimates from web search results. "
                "Return a short answer with a ballpark estimate and mention the currency.",
            ),
            (
                "human",
                "User request: {request}\n\nSearch results:\n{results}",
            ),
        ]
    )
    response = llm.invoke(prompt.format_messages(request=request, results=sources))

    return {
        "request": request,
        "estimate": str(response.content),
        "sources": sources,
    }
