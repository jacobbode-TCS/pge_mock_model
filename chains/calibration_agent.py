import os

from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from llm.llm import llm

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None


def double_check(request: str, max_results: int = 5) -> dict:
    """Double check the esimation agent's work to see if it's correct"""
    if not TAVILY_API_KEY or tavily_client is None:
        return {
            "request": request,
            "guidance": "Tavily API key is not configured.",
            "sources": [],
        }

    search_results = tavily_client.search(
        query=f"estimated cost for {request} including hazards and unexpected costs",
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
                "You double-check the estimated cost provided by other agents. "
                "Point out any discrepancies and mention unexpected costs that could arise.",
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
        "guidance": str(response.content),
        "sources": sources,
    }