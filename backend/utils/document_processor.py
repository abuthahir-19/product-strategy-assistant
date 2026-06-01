import os
import sys
import tempfile
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Load, chunk and tag uploaded documents."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def process_file(
        self,
        content: bytes,
        filename: str,
        doc_type: str = "general",
    ) -> List[Document]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            raw_docs = self._load(tmp_path, ext)
        finally:
            os.unlink(tmp_path)

        for doc in raw_docs:
            doc.metadata.update({"source": filename, "doc_type": doc_type})

        return self._splitter.split_documents(raw_docs)

    def process_text(
        self,
        text: str,
        source: str = "manual",
        doc_type: str = "general",
    ) -> List[Document]:
        doc = Document(
            page_content=text,
            metadata={"source": source, "doc_type": doc_type},
        )
        return self._splitter.split_documents([doc])

    # ------------------------------------------------------------------ #
    #  Private loader                                                      #
    # ------------------------------------------------------------------ #
    def _load(self, path: str, ext: str) -> List[Document]:
        if ext == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(path).load()

        if ext == "csv":
            from langchain_community.document_loaders import CSVLoader
            return CSVLoader(path, encoding="utf-8").load()

        if ext in ("docx", "doc"):
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(path).load()

        # Default: plain text / markdown / json
        from langchain_community.document_loaders import TextLoader
        return TextLoader(path, encoding="utf-8").load()
