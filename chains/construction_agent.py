import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from llm.llm import llm

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None


def provide_construction_guidance(request: str, max_results: int = 5) -> dict:
    """Look up assembly or usage guidance for a model or product mentioned by the user."""
    if not TAVILY_API_KEY or tavily_client is None:
        return {
            "request": request,
            "guidance": "Tavily API key is not configured.",
            "sources": [],
        }

    search_results = tavily_client.search(
        query=f"assembly instructions usage guide for {request}",
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
                "You provide practical guidance for assembling or using a product or model. "
                "Summarize the key steps and highlight any safety cautions.",
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
