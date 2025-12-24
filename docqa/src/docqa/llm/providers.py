import os
from docqa.config import Settings


def make_llm(settings: Settings):
    """
    Returns a chat model instance based on Settings.
    Supported: ollama, openai
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("DOCQA_OPENAI_API_KEY is required when llm_provider=openai")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unsupported llm_provider={provider}")


def make_embeddings(settings: Settings):
    """
    Returns an embeddings model instance based on Settings.
    Supported: ollama, openai
    """
    provider = settings.embed_provider.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=settings.embed_model)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise ValueError("DOCQA_OPENAI_API_KEY is required when embed_provider=openai")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        return OpenAIEmbeddings(model=settings.embed_model)

    raise ValueError(f"Unsupported embed_provider={provider}")
