import pytest
import tempfile
from pathlib import Path
from docqa.config import Settings
from docqa.pipeline.engine import QAEngine


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def isolated_engine(self):
        """Create an isolated QAEngine instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                llm_provider="ollama",
                embed_provider="ollama",
                llm_model="qwen2.5:7b",
                embed_model="nomic-embed-text",
                faiss_index_dir=str(Path(temp_dir) / "faiss_index"),
                chunk_size=500,
                chunk_overlap=50,
            )
            engine = QAEngine(settings=settings)
            yield engine

    def test_engine_ingest_and_answer(self, isolated_engine, sample_json_file):
        """Test ingesting and answering with isolated engine."""
        result = isolated_engine.ingest_json(sample_json_file)
        assert result["chunks_added"] > 0

        answer = isolated_engine.answer("Which cloud provider is used?")
        assert "answer" in answer
        assert "sources" in answer
        assert answer["question"] == "Which cloud provider is used?"

    def test_engine_batch_answer(self, ingested_engine):
        """Test batch answering with isolated engine (pre-ingested)."""
        questions = [
            "What is the company name?",
            "Which cloud provider is used?",
            "What is the main product?"
        ]

        answers = []
        for question in questions:
            answer = ingested_engine.answer(question)
            answers.append(answer)

        assert len(answers) == 3
        for answer in answers:
            assert "answer" in answer
            assert "sources" in answer
            assert "question" in answer

    def test_engine_answer_mapping(self, ingested_engine):
        """Test answering multiple questions and mapping results (pre-ingested)."""
        questions = [
            "What is the company name?",
            "Which cloud provider is used?"
        ]

        answers_list = []
        for question in questions:
            answer = ingested_engine.answer(question)
            answers_list.append(answer)

        assert len(answers_list) == 2
