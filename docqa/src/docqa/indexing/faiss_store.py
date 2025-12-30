from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def load_faiss(index_dir: str, embeddings) -> Optional[FAISS]:
    if not Path(index_dir).exists():
        return None

    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def save_faiss(store: FAISS, index_dir: str) -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(index_dir)



def add_documents_to_faiss(
    store: Optional[FAISS],
    docs: Iterable[Document],
    embeddings,
    *,
    index_dir: Optional[str] = None,
) -> FAISS:
    """
    Incrementally add documents to an existing store.
    If store is None, create a new one from docs.
    Optionally persist to disk if index_dir is provided.
    """
    docs_list = list(docs)
    if not docs_list:
        if store is None:
            raise ValueError("No documents to add and store is None.")
        return store

    if store is None:
        store = FAISS.from_documents(docs_list, embeddings)
    else:
        store.add_documents(docs_list)

    if index_dir:
        save_faiss(store, index_dir)

    return store
