from typing import List, Optional, Tuple

from langchain_core.documents import Document
from docqa.config import Settings


def _distance_to_relevance(distance: float) -> float:
    """
    FAISS similarity_search_with_score returns a distance (lower is better).
    Convert to a rough relevance score in (0, 1].
    """
    return 1.0 / (1.0 + float(distance))


def retrieve(
    vector_store,
    query: str,
    settings: Settings,
) -> Tuple[List[Document], Optional[List[float]]]:
    """
    Returns (docs, scores) where scores are OPTIONAL and represent a relevance-like score
    (higher is better). For MMR, scores are typically not available.
    """
    rtype = settings.retrieval_type
    k = settings.retrieval_k

    if rtype == "similarity":
        pairs = vector_store.similarity_search_with_score(query, k=k)
        docs = [d for d, _ in pairs]
        scores = [_distance_to_relevance(s) for _, s in pairs]
        return docs, scores

    if rtype == "mmr":
        # Diverse retrieval, usually better for long documents
        docs = vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=settings.retrieval_fetch_k,
            lambda_mult=settings.retrieval_lambda_mult,
        )
        return docs, None

    if rtype == "similarity_score_threshold":
        # Implement threshold using FAISS distances -> relevance conversion
        pairs = vector_store.similarity_search_with_score(query, k=k)
        filtered_docs: List[Document] = []
        filtered_scores: List[float] = []
        for d, dist in pairs:
            rel = _distance_to_relevance(dist)
            if rel >= float(settings.score_threshold):
                filtered_docs.append(d)
                filtered_scores.append(rel)
        return filtered_docs, filtered_scores

    raise ValueError(f"Unknown retrieval_type={rtype}")
