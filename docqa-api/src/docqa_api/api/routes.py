import json
import os
import tempfile
from typing import Any, Dict

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from docqa.pipeline.engine import QAEngine
from .deps import get_engine

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _save_upload_to_temp(upload: UploadFile) -> str:
    """Save UploadFile to a temp file and return the file path."""
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.rsplit(".", 1)[-1].lower()

    fd, path = tempfile.mkstemp(prefix="docqa_", suffix=suffix)
    os.close(fd)

    with open(path, "wb") as f:
        f.write(upload.file.read())

    return path


@router.post("/ingest")
def ingest(file: UploadFile = File(...), engine: QAEngine = Depends(get_engine)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    tmp = _save_upload_to_temp(file)
    try:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        fn = {"pdf": engine.ingest_pdf, "json": engine.ingest_json}.get(ext)
        if not fn:
            raise HTTPException(400, "Upload a .pdf or .json file")
        return {"status": "ok", "ingest": fn(tmp), "file_type": ext}
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


@router.post("/answer")
def answer(
    payload: Dict[str, Any] = Body(...),
    engine: QAEngine = Depends(get_engine),
) -> Dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Request body must be JSON with key 'question' as a non-empty string.",
        )

    return engine.answer(question.strip())


@router.post("/answer/batch")
def answer_batch(
    file: UploadFile = File(...),
    engine: QAEngine = Depends(get_engine),
):
    if not (file.filename or "").endswith(".json"):
        raise HTTPException(400, "Upload a .json file")

    data = json.loads(file.file.read().decode())
    qs = data if isinstance(data, list) else data["questions"]

    return {
        q.strip(): engine.answer(q.strip())["answer"]
        for q in qs
        if isinstance(q, str) and q.strip()
    }
