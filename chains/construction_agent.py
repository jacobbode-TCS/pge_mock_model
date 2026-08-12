import os

from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from llm.llm import llm

SKIP_LLM = False  # Set to True to skip LLM processing for testing purposes

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

    # Skip the LLM for testing purposes
    if SKIP_LLM:
        return {
            "request": request,
            "guidance": "LLM processing is skipped.",
            "sources": sources,
        }

    # Resume LLM processing
    system_text = (
        "You provide practical guidance for assembling or using a product or model. "
        "Summarize the key steps and highlight any safety cautions."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", "User request: {request}\n\nSearch results:\n{results}"),
        ]
    )
    response = llm.invoke(prompt.format_messages(request=request, results=sources))
    print(f"Tool calls: {response.tool_calls}")

    return {
        "request": request,
        "guidance": str(response.content),
        "sources": sources,
    }
