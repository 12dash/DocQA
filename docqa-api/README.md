# DocQA API

FastAPI service that exposes the DocQA core library as a REST API for
document ingestion and question answering.

## Overview

This project is a thin HTTP wrapper around the **DocQA** core package.  
It provides endpoints for:

- Ingesting documents (PDF, JSON)
- Answering questions using retrieved context
- Batch question answering (challenge-compatible output)

All retrieval and LLM logic lives in the `docqa` core package.  
This API layer is responsible only for HTTP, validation, and I/O.

## Installation

### Prerequisites
The project uses poetry to manage the dependencies. Please install poetry and then follow to the next step to install the dependencies.
- Poetry installed (`pip install poetry`)

### Install dependencies
From the **docqa-api** project root:

```bash
poetry install
```

This installs all the necessary packages for running the Fast API.

## Run the API 
```bash
poetry run uvicorn docqa_api.api.main:app --reload
```