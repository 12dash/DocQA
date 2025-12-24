import pytest
from docqa.pipeline.engine import QAEngine
from langchain_core.documents import Document


@pytest.mark.unit
class TestQAEngine:
    """Unit tests for QAEngine."""

    def test_qa_engine_initialization(self, qa_engine):
        """Test QAEngine initialization."""
        assert qa_engine is not None
        assert qa_engine.settings is not None
        assert qa_engine.llm is not None
        assert qa_engine.embeddings is not None

    def test_ingest_documents(self, qa_engine):
        """Test document ingestion."""
        docs = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "wiki", "page": 1}
            ),
            Document(
                page_content="FastAPI is a modern web framework.",
                metadata={"source": "docs", "page": 1}
            ),
        ]
        
        result = qa_engine.ingest_documents(docs)
        assert result["chunks_added"] == 2

    def test_ingest_empty_documents_raises(self, qa_engine):
        """Ingesting an empty list should raise ValueError when store is None."""
        with pytest.raises(ValueError, match="No documents to add and store is None"):
            qa_engine.ingest_documents([])

    def test_document_with_rich_metadata(self, qa_engine):
        """Test ingesting document with rich metadata."""
        docs = [
            Document(
                page_content="Document content here.",
                metadata={
                    "source": "test.pdf",
                    "page": 5,
                    "author": "Test Author"
                }
            )
        ]
        
        result = qa_engine.ingest_documents(docs)
        assert result["chunks_added"] == 1