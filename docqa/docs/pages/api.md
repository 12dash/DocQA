# API 
This document describes the REST API for the **Document Question-Answering service**.  
The API allows users to upload documents, index them into a searchable knowledge base, and retrieve answers grounded strictly in the document content.

## Endpoints

### POST `/upload-document`
Performs asynchronous ingestion and indexing; supports a synchronous mode for small files via a query parameter (e.g. ?sync=true).

- Accepts multipart/form-data with a file (PDF or JSON).
- Returns: {}"doc_id": "<uuid>", "ingest_status": "queued"}.

### GET `/status/{doc_id}`
Returns ingest/indexing status, queued tasks, and basic metadata.

### GET `/docs/{doc_id}/metadata`
Returns stored metadata, file hash, vector index statistics and ingest timestamps.

### POST `/questions`
Answer a user question by retrieving most relevant chunks from the vector db and framing the answer using LLM.

- Accepts JSON: { "doc_id": "<uuid>", "questions": ["q1", "q2", ...], "retrieval": { "top_k": 5 } } or a JSON file upload of questions.
- Returns structured JSON mapping each question to answer, sources and confidence scores.

## Example Output 

Following is the output for each question provided in the json.

```json
{
	"question": "Which cloud providers do you rely on?",
	"answer": "We rely on Amazon Web Services (primary) and Google Cloud Platform (backup), as documented on page 12.",
	"sources": [
		{ "doc_id": "abc", "page": 12, "chunk_index": 3, "text_snippet": "...we run on AWS...", "score": 0.92 }
	],
	"confidence": 0.87,
	"model": "gpt-4o-mini"
}
```
