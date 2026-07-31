import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY1", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL1", "http://127.0.0.1:1234/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL1", "local-model")

# Build LLM using the variables above
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model=OPENAI_MODEL,
)