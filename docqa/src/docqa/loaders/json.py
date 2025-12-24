import json
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_core.documents import Document


def _read_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")

    raw = p.read_text(encoding="utf-8-sig")
    return json.loads(raw)


def load_json(
    path: Union[str, Path],
    *,
    include_answer: bool = True,
    include_comments: bool = True,
) -> List[Document]:
    """
    Load a JSON knowledge base as LangChain Documents.

    Each record becomes one Document with rich text:
      Question: ...
      Answer: ...
      Comments: ...

    Metadata includes:
      - source, source_type
    """
    p = Path(path)
    data = _read_json(p)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array (list).")

    docs: List[Document] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected list items to be objects (dict). Got {type(item)} at index {i}")

        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        c = (item.get("comments") or "").strip()

        parts: List[str] = []
        if q: parts.append(f"Question: {q}")
        if include_answer and a: parts.append(f"Answer: {a}")
        if include_comments and c: parts.append(f"Comments: {c}")

        text = "\n".join(parts).strip()
        if not text: continue

        metadata: Dict[str, Any] = {
            "source": str(p),
            "source_type": "json",
            "row_index": i,
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs
