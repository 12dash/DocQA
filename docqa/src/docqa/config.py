from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------
    # Providers
    # -----------------------
    llm_provider: str = Field(default="ollama", description="ollama | openai")
    embed_provider: str = Field(
        default="ollama", description="ollama | openai")

    # -----------------------
    # Models
    # -----------------------
    llm_model: str = Field(default="qwen2.5:7b",
                           description="e.g. qwen2.5:7b or gpt-4o-mini")
    llm_temperature: float = Field(
        default=0.0, description="0.0 for deterministic QA")
    llm_num_ctx: int = Field(
        default=4096, description="Ollama context window tokens (if supported)")

    embed_model: str = Field(default="nomic-embed-text",
                             description="e.g. nomic-embed-text or text-embedding-3-small")

    # -----------------------
    # Chunking
    # -----------------------
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # -----------------------
    # Retrieval
    # -----------------------
    retrieval_type: str = Field(
        default="similarity", description="similarity | mmr | similarity_score_threshold")
    retrieval_k: int = Field(
        default=100, description="How many chunks to return")
    retrieval_fetch_k: int = Field(
        default=50, description="MMR only: candidates to fetch before reranking")
    retrieval_lambda_mult: float = Field(
        default=0.5, description="MMR only: 0=more diverse, 1=more relevant")
    score_threshold: float = Field(
        default=0.0, description="Only for similarity_score_threshold")

    # -----------------------
    # Vector store persistence
    # -----------------------
    faiss_index_dir: str = Field(default="./.local/faiss_index")

    # -----------------------
    # OpenAI
    # -----------------------
    openai_api_key: str | None = Field(default=None)

    def validate(self) -> None:
        allowed = {"ollama", "openai"}
        if self.llm_provider not in allowed:
            raise ValueError(
                f"Invalid llm_provider={self.llm_provider}. Allowed: {allowed}")
        if self.embed_provider not in allowed:
            raise ValueError(
                f"Invalid embed_provider={self.embed_provider}. Allowed: {allowed}")
        if self.retrieval_type not in {"similarity", "mmr", "similarity_score_threshold"}:
            raise ValueError(
                "Invalid retrieval_type. Allowed: similarity | mmr | similarity_score_threshold"
            )
        if self.retrieval_k <= 0:
            raise ValueError("retrieval_k must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
