import pytest
from docqa.retrieval.retriever import _distance_to_relevance


@pytest.mark.unit
class TestRetriever:
    """Unit tests for retrieval functions."""

    def test_distance_to_relevance_zero_distance(self):
        """Test conversion of zero distance to relevance."""
        relevance = _distance_to_relevance(0.0)
        
        assert relevance == 1.0

    def test_distance_to_relevance_large_distance(self):
        """Test conversion of large distance to relevance."""
        relevance = _distance_to_relevance(1.0)
        
        assert 0 < relevance < 1.0
        assert abs(relevance - 0.5) < 0.01  # Should be close to 0.5

    def test_distance_to_relevance_monotonic(self):
        """Test that relevance decreases as distance increases."""
        relevance_1 = _distance_to_relevance(0.5)
        relevance_2 = _distance_to_relevance(1.0)
        relevance_3 = _distance_to_relevance(2.0)
        
        assert relevance_1 > relevance_2
        assert relevance_2 > relevance_3

    def test_distance_to_relevance_bounds(self):
        """Test that relevance is always in (0, 1]."""
        for distance in [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]:
            relevance = _distance_to_relevance(distance)
            assert 0 < relevance <= 1.0