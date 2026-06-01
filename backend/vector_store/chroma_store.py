import os
import shutil
import sys
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PERSIST_DIR, MAX_RETRIEVAL_DOCS

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ------------------------------------------------------------------ #
#  Local embedding — uses chromadb's built-in ONNX MiniLM model.     #
#  No API key or network call required; runs entirely on-device.      #
#  The gateway is used only for LLM inference (agents + chat).        #
# ------------------------------------------------------------------ #

class _LocalEmbeddings(Embeddings):
    """Thin LangChain wrapper around chromadb's DefaultEmbeddingFunction."""

    _fn = None  # shared across all instances

    @classmethod
    def _get_fn(cls):
        if cls._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            cls._fn = DefaultEmbeddingFunction()
        return cls._fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        fn = self._get_fn()
        # .tolist() converts np.float32 → Python float, required by chromadb ≥0.5
        return [np.array(vec).tolist() for vec in fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        import numpy as np
        fn = self._get_fn()
        return np.array(fn([text])[0]).tolist()


# ------------------------------------------------------------------ #
#  Vector store                                                        #
# ------------------------------------------------------------------ #

class ChromaVectorStore:
    """Singleton ChromaDB vector store backed by local ONNX embeddings."""

    _instance: "ChromaVectorStore | None" = None

    def __new__(cls) -> "ChromaVectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return
        self._embeddings = _LocalEmbeddings()
        self._persist_dir = CHROMA_PERSIST_DIR
        self._db: Optional[Chroma] = None
        self._ready = True

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _get_db(self) -> Chroma:
        if self._db is None:
            self._db = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embeddings,
                collection_name="product_strategy",
            )
        return self._db

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Document]) -> None:
        self._get_db().add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        k = k or MAX_RETRIEVAL_DOCS
        db = self._get_db()
        if filter_dict:
            return db.similarity_search(query, k=k, filter=filter_dict)
        return db.similarity_search(query, k=k)

    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        doc_type: Optional[str] = None,
    ) -> str:
        """Return a single string of joined document chunks."""
        try:
            filter_dict = {"doc_type": doc_type} if doc_type else None
            docs = self.similarity_search(query, k=k, filter_dict=filter_dict)
            return "\n\n---\n\n".join(d.page_content for d in docs)
        except Exception:
            return ""

    def as_retriever(self, **kwargs):
        return self._get_db().as_retriever(**kwargs)

    def get_document_count(self) -> int:
        try:
            return self._get_db()._collection.count()
        except Exception:
            return 0

    def has_documents(self) -> bool:
        return self.get_document_count() > 0

    def clear(self) -> None:
        if self._db is not None:
            try:
                self._db.delete_collection()
            except Exception:
                pass
            self._db = None
        # Release SQLite/ONNX file handles (important on Windows)
        import gc
        gc.collect()
        if os.path.exists(self._persist_dir):
            try:
                shutil.rmtree(self._persist_dir)
            except PermissionError:
                # Retry once after a brief pause for Windows file-lock to release
                import time
                time.sleep(0.3)
                try:
                    shutil.rmtree(self._persist_dir)
                except Exception:
                    pass  # Directory will be reused cleanly on next init
