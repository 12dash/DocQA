from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.documents import Document

from docqa.config import Settings
from docqa.chunking import split_documents
from docqa.indexing.faiss_store import load_faiss, add_documents_to_faiss, save_faiss
from docqa.llm.providers import make_llm, make_embeddings
from docqa.llm.prompts import build_grounded_prompt
from docqa.retrieval.retriever import retrieve
from docqa.loaders.pdf import load_pdf
from docqa.loaders.json import load_json

PathLike = Union[str, Path]


class QAEngine:
    """
    Core RAG engine:
    - ingest docs into a persistent FAISS store
    - retrieve + answer with an LLM
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.settings.validate()

        self._lock = RLock()

        self.embeddings = make_embeddings(self.settings)
        self.llm = make_llm(self.settings)
        self.vector_store = load_faiss(self.settings.faiss_index_dir, self.embeddings)

    def ingest_documents(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Split and add docs to the vector store. Persists to disk.
        """
        with self._lock:
            splits = split_documents(
                docs,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )

            for i, d in enumerate(splits):
                d.metadata = d.metadata or {}
                d.metadata.setdefault("chunk_index", i)

            self.vector_store = add_documents_to_faiss(
                self.vector_store,
                splits,
                self.embeddings,
                index_dir=self.settings.faiss_index_dir,
            )

            return {
                "ingested_pages": len(docs),
                "chunks_added": len(splits),
                "index_dir": self.settings.faiss_index_dir,
            }

    def ingest_pdf(self, pdf_path: PathLike) -> Dict[str, Any]:
        docs = load_pdf(pdf_path)
        return self.ingest_documents(docs)

    def ingest_json(self, json_path: PathLike) -> Dict[str, Any]:
        docs = load_json(json_path)
        return self.ingest_documents(docs)

    def _build_context(self, docs_and_scores: Sequence[Tuple[Document, float]]) -> str:
        """
        Build context from retrieved sources by appending metadata and citations for the downstream LLM.
        """
        parts: List[str] = []

        for i, (doc, score) in enumerate(docs_and_scores, 1):
            meta = doc.metadata or {}
            cite = []

            if meta.get("source_type") == "pdf":
                if "page" in meta:
                    cite.append(f"page={meta['page']}")

            if meta.get("source_type") == "json" and meta.get("id"):
                cite.append(f"id={meta['id']}")

            cite_str = ", ".join(cite) if cite else "no-meta"
            chunk = doc.page_content or ""
            
            if not chunk:
                continue
            chunk = chunk.strip()
            if not chunk:
                continue

            formatted_chunk = f"[{i}] ({cite_str}, score={score:.4f})\n{chunk}"
            parts.append(formatted_chunk)

        return "\n\n".join(parts)

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a single question. Returns a JSON-serializable dict.
        """
        with self._lock:
            if self.vector_store is None:
                return {
                    "question": question,
                    "answer": self.settings.not_found_token,
                    "sources": [],
                    "model": getattr(self.llm, "model", None),
                }

            docs, scores = retrieve(self.vector_store, question, self.settings)

            if not docs:
                return {
                    "question": question,
                    "answer": self.settings.not_found_token,
                    "sources": [],
                    "model": getattr(self.llm, "model", None),
                }

            # Combine docs with scores for context building
            docs_and_scores: List[Tuple[Document, float]] = [
                (doc, score if scores and i < len(scores) else 0.0)
                for i, doc in enumerate(docs)
            ]

            context = self._build_context(docs_and_scores)
            prompt = build_grounded_prompt(
                context=context,
                question=question,
            )

            resp = self.llm.invoke(prompt)
            answer_text = (getattr(resp, "content", None) or str(resp)).strip()

            sources: List[Dict[str, Any]] = []
            for i, (doc, score) in enumerate(docs_and_scores):
                md = doc.metadata or {}
                sources.append(
                    {
                        "source": md.get("source"),
                        "source_type": md.get("source_type"),
                        "page": md.get("page"),
                        "chunk_index": md.get("chunk_index", i),
                        "doc_id": md.get("id"),
                        "text_snippet": (doc.page_content or "").replace("\n", " ").strip(),
                        "score": score,
                    }
                )

            return {
                "question": question,
                "answer": answer_text if answer_text else self.settings.not_found_token,
                "sources": sources,
                "model": getattr(self.llm, "model", None),
            }