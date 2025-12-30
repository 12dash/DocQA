import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from docqa_api.api.main import app
from docqa.config import Settings
from docqa.pipeline.engine import QAEngine


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_json_file(test_data_dir):
    """Return path to static sample JSON file."""
    json_file = test_data_dir / "sample.json"
    assert json_file.exists(), f"Test file not found: {json_file}"
    return str(json_file)


@pytest.fixture
def sample_json_file_temp(temp_dir):
    """Create a temporary sample JSON file for testing."""
    data = [
        {
            "id": "q1",
            "question": "What is the company name?",
            "answer": "Product Fruits",
            "comments": "Found in documentation"
        },
        {
            "id": "q2",
            "question": "Which cloud provider is used?",
            "answer": "AWS",
            "comments": "Primary infrastructure"
        }
    ]
    file_path = Path(temp_dir) / "sample.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(file_path)


@pytest.fixture
def test_settings(temp_dir):
    """Return test settings with temporary storage."""
    return Settings(
        llm_provider="ollama",
        embed_provider="ollama",
        llm_model="qwen2.5:7b",
        embed_model="nomic-embed-text",
        faiss_index_dir=str(Path(temp_dir) / "faiss_index"),
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture
def qa_engine(test_settings):
    """Return QAEngine instance for testing."""
    return QAEngine(settings=test_settings)


@pytest.fixture
def client():
    """FastAPI TestClient fixture for API tests."""
    return TestClient(app)


@pytest.fixture
def post_batch(client):
    """Helper to post batch question files to the API."""
    def _post(data):
        return client.post("/answer/batch", files={"file": ("test.json", json.dumps(data))})
    return _post


@pytest.fixture
def ingested_engine(qa_engine, sample_json_file):
    """Return a QAEngine with `sample_json_file` already ingested."""
    qa_engine.ingest_json(sample_json_file)
    return qa_engine