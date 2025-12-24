from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: List[Document],
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
    add_start_index: bool = True,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=add_start_index,
    )
    return splitter.split_documents(docs)
