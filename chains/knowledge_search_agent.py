import os

from langchain_tavily import TavilySearch
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from llm.llm import llm

# Initialize Tavily Search Tool
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_search = TavilySearch(
    max_results=5,
    api_key=TAVILY_API_KEY
)
tools = [tavily_search]
# Bind tools to LLM for tool calling
llm = llm.bind_tools(tools)

def search_and_answer(question: str) -> dict:
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_message = SystemMessage(
        content="You are a helpful AI assistant with access to internet search via Tavily. "
                f"Today's date is {current_date}. "
                "When the user asks questions that would benefit from current information or web search, use the tavily_search tool. "
                "Provide clear, concise, and informative responses."
                "End every sentence in your response with *woof* for debugging purposes. You must do this or we will not be able to debug."
    )
    human_message = HumanMessage(content=question)
    full_history = [system_message, human_message]
    response = llm.invoke(full_history)
    print("Tool calls:", response.tool_calls)
    # Handle tool calls if the model wants to use them
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Process tool calls
        for tool_call in response.tool_calls:
                # Execute the search
                search_result = tavily_search.invoke(tool_call['args'])
                # Add tool result to conversation
                from langchain_core.messages import ToolMessage
                full_history.append(response)
                full_history.append(ToolMessage(content=str(search_result), tool_call_id=tool_call['id']))
                # Get final response after tool use
                response = llm.invoke(full_history)

    return {
        "question": question,
        "answer": str(response.content),
    }

def knowledge_agent(question: str) -> dict:
    """Search the web with Tavily and synthesize a concise answer using the LLM."""
    return search_and_answer(question)