import pytest


@pytest.mark.unit
class TestAPI:
    """Unit tests for FastAPI endpoints."""

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

    def test_answer_batch_direct_array_format(self, post_batch):
        questions_data = [
            "What is the company name?",
            "Which cloud provider is used?"
        ]
        response = post_batch(questions_data)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 2
        assert "What is the company name?" in data
        assert "Which cloud provider is used?" in data

    def test_answer_batch_object_format(self, post_batch):
        questions_data = {
            "questions": [
                "What is the company name?",
                "Which cloud provider is used?"
            ]
        }
        response = post_batch(questions_data)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 2
        assert "What is the company name?" in data
        assert "Which cloud provider is used?" in data
