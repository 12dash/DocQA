# DocQA

**DocQA** is a document-grounded Question Answering API. Upload a **PDF or JSON** as a knowledge base and ask questions to receive **answers strictly derived from the document**, with traceable sources.
It is designed for **automation, compliance (GRC), and knowledge extraction** workflows.

## What it does
1. **Ingest documents**  
   Upload PDF or JSON files and index them into a vector database.
2. **Ask questions**  
   Submit a single question or a batch of questions as JSON.
3. **Get grounded answers**  
   Responses are generated using a RAG pipeline and are based only on retrieved document chunks.

## Users
- **Developer** – uploads documents and manages the knowledge base  
- **Consumer** – submits questions as JSON and receives answers

## How it works 
- Text is **chunked** with configurable size and overlap
- Embeddings are generated using **OpenAI** model.
- Vectors are stored in **FAISS**
- **Top-k semantic retrieval** is used per question
- Answers are generated with a **RAG pipeline** and cite source chunks
