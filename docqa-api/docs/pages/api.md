# API Endpoints
| Category | Method | Endpoint | Description |
|--------|--------|----------|-------------|
| Health Check | GET | `/health` | Returns API health status |
| Document Management | POST | `/ingest` | Upload and index PDF or JSON documents |
| Question Answering | POST | `/answer` | Answer a single question |
| Question Answering | POST | `/answer/batch` | Upload JSON file with questions array or {"questions": [...]} |

## Swagger Testing
Run the following command from `docqa-api` 
```bash
poetry run uvicorn docqa_api.api.main:app --reload
``` 

Then visit `http://localhost:8000/docs` to interact with the API.

You will be able to view and interact with the API 

![Swagger Page](../img/swaggerpage.png)

### Ingest
We can ingest a PDF file and check the output.
![Ingest PDF](../img/ingest-input.png)

The API returned a 200 OK response, indicating that the file was successfully processed and added to the FAISS vector database.
![Ingest Response](../img/ingest-response.png)

### Answer 
There are 2 APIs for answering a question. The first one is mostly a test api which returns the answer along with other metadata information about the retreval process such as the chunk index, process score etc. 

For the current use case we will look at the `/answer_batch` api which returns the answers to a batch of questions asked in the form of a json file. 
![Answer Input](../img/answer-batch-input.png)

The response is in the form of a question, answer pair
![Answer Response](../img/answer-batch-response.png)