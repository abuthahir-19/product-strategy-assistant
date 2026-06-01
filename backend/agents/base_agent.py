"""Base agent: shared LLM client, vector-store access, and call helper."""
import os
import sys
import warnings
from typing import Optional

# Suppress SSL certificate verification warnings produced by httpx/urllib3
# when connecting to the course gateway (self-signed cert).
warnings.filterwarnings("ignore", message=".*Unverified HTTPS.*")
warnings.filterwarnings("ignore", message=".*InsecureRequestWarning.*")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from vector_store.chroma_store import ChromaVectorStore

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Shared httpx client that skips SSL cert verification for the course gateway
_HTTP_CLIENT = httpx.Client(verify=False)


class BaseAgent:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            http_client=_HTTP_CLIENT,
        )
        self._store = ChromaVectorStore()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        k: int = 6,
        doc_type: Optional[str] = None,
    ) -> str:
        """Return joined text chunks from the vector store."""
        return self._store.get_context(query, k=k, doc_type=doc_type)

    def llm_call(self, system_prompt: str, user_content: str) -> str:
        response = self._llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return response.content

    # ------------------------------------------------------------------ #
    #  Override in subclasses                                              #
    # ------------------------------------------------------------------ #
    def run(self, state: dict) -> dict:
        raise NotImplementedError(f"{self.name}.run() not implemented")
