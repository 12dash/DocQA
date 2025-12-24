from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: Union[str, Path]) -> List[Document]:
    """
    Load a PDF file into LangChain Document objects.

    Notes:
    - PyPDFLoader returns one Document per page by default.
    - Metadata typically includes page number, source, etc.
    """
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"PDF not found: {p}")

    loader = PyPDFLoader(str(p))
    docs = loader.load()

    for d in docs:
        d.metadata.setdefault("source", str(p))
        d.metadata.setdefault("source_type", "pdf")

        if "page" not in d.metadata and "page_number" in d.metadata:
            d.metadata["page"] = d.metadata["page_number"]

    return docs
