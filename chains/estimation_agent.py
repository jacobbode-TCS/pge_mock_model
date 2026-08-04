import os

from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from llm.llm import llm

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

    system_text = (
        "You provide rough cost estimates from web search results. "
        "Return a short answer with a ballpark estimate and mention the currency."
    )
    human_text = f"User request: {request}\n\nSearch results:\n{sources}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", "User request: {request}\n\nSearch results:\n{results}"),
        ]
    )
    response = llm.invoke(prompt.format_messages(request=request, results=sources))

    return {
        "request": request,
        "estimate": str(response.content),
        "sources": sources,
    }
