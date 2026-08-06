import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "local-model")

# Build LLM using the variables above
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model=OPENAI_MODEL,
)


class Conversation:
    """Lightweight conversation wrapper that keeps message history and
    forwards the accumulated messages to the underlying `llm.invoke()`.

    Usage:
        conv = Conversation()  # uses module-level `llm` by default
        conv.add_system("You are helpful")
        conv.add_user("Hi")
        resp = conv.invoke()
    """

    def __init__(self, llm_client=None) -> None:
        self.llm = llm_client or llm
        self.history: list = []

    def add_system(self, content: str) -> None:
        self.history.append(SystemMessage(content=content))

    def add_user(self, content: str) -> None:
        self.history.append(HumanMessage(content=content))

    def add_assistant(self, content: str) -> None:
        self.history.append(AIMessage(content=content))

    def invoke(self, extra_messages: list | None = None):
        """Invoke the LLM with the accumulated history plus any extra messages.

        If `extra_messages` is provided it should be a list of langchain_core
        message objects or a list-compatible value accepted by the LLM wrapper.
        The assistant reply is appended to history (if present in response).
        """
        messages = list(self.history)
        if extra_messages:
            messages.extend(extra_messages)

        response = self.llm.invoke(messages)

        # Normalise common response shapes and append assistant content to history
        content = None
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict) and "content" in response:
            content = response["content"]

        if content is not None:
            try:
                # store as AssistantMessage for future context
                self.add_assistant(str(content))
            except Exception:
                pass

        return response


def create_conversation(llm_client=None) -> Conversation:
    return Conversation(llm_client=llm_client)