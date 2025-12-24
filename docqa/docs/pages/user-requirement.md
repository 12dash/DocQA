# User Requirements 

This API is a document-based Question Answering service that allows users to upload a document (PDF or JSON) that serves as a knowledge base. The API processes the document, converts its content into searchable embeddings using a vector database, and answers each question strictly based on the document’s information. It returns the results as a JSON object mapping each question to its corresponding answer, enabling automated analysis, compliance checks, and knowledge extraction from documents.

## User Roles

User roles refers to the people who would be interacting with the API. We will define 2 roles
- `Developer`: A user who uploads the documents to form the knowledge base. 
- `Consumer`: A user who would be asking question by passing the json objects. 

## Functional Requirements

### 1. Inputs
- Accept **PDF** and **JSON** files only; reject all other formats with **HTTP 415**.
- Enforce a configurable **maximum file size** (default: 50 MB); return **HTTP 413** if exceeded.
- Extract and normalize text from PDF and JSON inputs.

### 2. Text Processing
- Chunk text using configurable **chunk size** and **overlap**.
- Preserve minimal metadata per chunk:
  - `doc_id`
  - `page` (if applicable)
  - `chunk_index`

### 3. Embeddings & Storage
- Generate embeddings using a configurable provider (`OpenAI`).
- Store embeddings in a Vector Database (`FAISS`).
- Persist chunk metadata alongside each vector for traceability.

### 4. Retrieval & Answering
- Perform **top-k semantic retrieval** per question.
- Generate answers using a **Retrieval-Augmented Generation (RAG)** pipeline with a large language model.
- Answers must be grounded strictly in retrieved content by citing the source.