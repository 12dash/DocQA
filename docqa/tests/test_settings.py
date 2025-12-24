import pytest
from docqa.config import Settings


@pytest.mark.unit
class TestSettings:
    """Unit tests for Settings configuration."""

    def test_settings_defaults(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.llm_provider == "ollama"
        assert settings.embed_provider == "ollama"
        assert settings.llm_temperature == 0.0
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0

    def test_settings_validate_success(self):
        """Test validation passes with valid settings."""
        settings = Settings(
            llm_provider="ollama",
            embed_provider="ollama"
        )
        
        # Should not raise
        settings.validate()

    def test_settings_validate_invalid_llm_provider(self):
        """Test validation fails with invalid LLM provider."""
        settings = Settings(llm_provider="invalid_provider")
        
        with pytest.raises(ValueError, match="Invalid llm_provider"):
            settings.validate()

    def test_settings_validate_invalid_embed_provider(self):
        """Test validation fails with invalid embed provider."""
        settings = Settings(embed_provider="invalid_provider")
        
        with pytest.raises(ValueError, match="Invalid embed_provider"):
            settings.validate()

    def test_settings_validate_invalid_retrieval_type(self):
        """Test validation fails with invalid retrieval type."""
        settings = Settings(retrieval_type="invalid_type")
        
        with pytest.raises(ValueError, match="Invalid retrieval_type"):
            settings.validate()

    def test_settings_validate_invalid_retrieval_k(self):
        """Test validation fails with invalid retrieval_k."""
        settings = Settings(retrieval_k=0)
        
        with pytest.raises(ValueError, match="retrieval_k must be > 0"):
            settings.validate()

    def test_settings_validate_invalid_chunk_size(self):
        """Test validation fails with invalid chunk_size."""
        settings = Settings(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            settings.validate()