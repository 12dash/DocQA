import os
import tempfile
from typing import Dict, Any

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Body

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
) -> Dict[str, str]:
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    if not file.filename.lower().endswith('.json'):
        raise HTTPException(400, "Upload a .json file")

    try:
        import json
        content = json.loads(file.file.read().decode('utf-8'))

        if isinstance(content, list):
            questions = content
        elif isinstance(content, dict) and "questions" in content:
            questions = content["questions"]
        else:
            raise HTTPException(
                status_code=400,
                detail="JSON file must be either an array of questions or an object with a 'questions' key containing an array.",
            )

        if not isinstance(questions, list) or not questions:
            raise HTTPException(
                status_code=400,
                detail="Questions must be a non-empty list of strings.",
            )

        cleaned: list[str] = []
        for q in questions:
            if not isinstance(q, str) or not q.strip():
                raise HTTPException(
                    status_code=400,
                    detail="All items in 'questions' must be non-empty strings.",
                )
            cleaned.append(q.strip())

        return {q: engine.answer(q)["answer"] for q in cleaned}

    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")
