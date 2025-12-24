import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from docqa_api.api.main import app
from docqa.config import Settings
from docqa.pipeline.engine import QAEngine


@pytest.mark.unit
class TestAPI:
    """Unit tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ingest_missing_file(self, client):
        response = client.post("/ingest")
        
        assert response.status_code == 422

    def test_answer_missing_payload(self, client):
        response = client.post("/answer", json={})
        assert response.status_code == 400

    def test_answer_batch_missing_file(self, client):
        response = client.post("/answer/batch")
        
        assert response.status_code == 422

    def test_answer_batch_invalid_file_type(self, client):
        response = client.post("/answer/batch", files={"file": ("test.txt", "not json content")})
        
        assert response.status_code == 400
        assert "Upload a .json file" in response.json()["detail"]

    def test_answer_batch_direct_array_format(self, client):
        questions_data = [
            "What is the company name?",
            "Which cloud provider is used?"
        ]
        import json
        response = client.post("/answer/batch", files={"file": ("test.json", json.dumps(questions_data))})
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 2
        assert "What is the company name?" in data
        assert "Which cloud provider is used?" in data

    def test_answer_batch_object_format(self, client):
        questions_data = {
            "questions": [
                "What is the company name?",
                "Which cloud provider is used?"
            ]
        }
        import json
        response = client.post("/answer/batch", files={"file": ("test.json", json.dumps(questions_data))})
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 2
        assert "What is the company name?" in data
        assert "Which cloud provider is used?" in data


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
        # Ingest the document
        result = isolated_engine.ingest_json(sample_json_file)
        
        assert result["chunks_added"] > 0
        
        # Ask a question
        answer = isolated_engine.answer("Which cloud provider is used?")
        
        assert "answer" in answer
        assert "sources" in answer
        assert answer["question"] == "Which cloud provider is used?"

    def test_engine_batch_answer(self, isolated_engine, sample_json_file):
        """Test batch answering with isolated engine."""
        isolated_engine.ingest_json(sample_json_file)
        
        questions = [
            "What is the company name?",
            "Which cloud provider is used?",
            "What is the main product?"
        ]
        
        # Answer each question individually
        answers = []
        for question in questions:
            answer = isolated_engine.answer(question)
            answers.append(answer)

        assert len(answers) == 3
        
        for answer in answers:
            assert "answer" in answer
            assert "sources" in answer
            assert "question" in answer

    def test_engine_answer_mapping(self, isolated_engine, sample_json_file):
        """Test answering multiple questions and mapping results."""
        # Ingest the document
        isolated_engine.ingest_json(sample_json_file)
        
        # Get answers for multiple questions
        questions = [
            "What is the company name?",
            "Which cloud provider is used?"
        ]
        
        answers_list = []
        for question in questions:
            answer = isolated_engine.answer(question)
            answers_list.append(answer)
        
        assert len(answers_list) == 2
