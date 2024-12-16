# LLM-Based Question-Answering Chatbot with RAG Pipeline

## Project Overview
This project implements a Question-Answering (QA) chatbot powered by a Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG) pipeline. The system is designed to deliver accurate and contextual responses by integrating an LLM with a vectorized document database for efficient retrieval.

---

## Key Features
- **LLM Model**: The system uses `transformers` library to leverage state-of-the-art language models like GPT or T5.
- **RAG Pipeline**: Combines document retrieval with LLM-based generation to provide relevant answers.
- **Vectorization Techniques**: Utilizes `sentence-transformers` to generate dense vector representations of text for efficient similarity search.
- **Vector Database**: Powered by `FAISS` for fast and scalable similarity search and document retrieval.

---

## Tools and Packages
### Main Dependencies
| Package                 | Description                                      |
|-------------------------|--------------------------------------------------|
| `transformers`          | Provides pre-trained LLMs for natural language processing. |
| `sentence-transformers` | Generates embeddings for text using state-of-the-art models. |
| `faiss`                 | An open-source library for vector similarity search. |
| `huggingface_hub`       | Facilitates downloading pre-trained models and tokenizers. |
| `numpy`                 | Supports numerical computations and array handling. |
| `pandas`                | Used for managing data in tabular format. |
| `uvicorn` and `FastAPI` | Builds and serves the RESTful API for the chatbot. |

### Additional Tools
| Tool/Library            | Purpose                                         |
|-------------------------|--------------------------------------------------|
| `pipdeptree`            | Dependency visualization and conflict resolution. |
| `dotenv`                | For managing environment variables securely. |
| `pytest`                | For testing the functionality of the pipeline. |
| `black` and `flake8`    | For code formatting and linting. |

---

## File Structure
```plaintext
.
├── src
│   ├── models
│   │   ├── rag_model.py
│   │       ├── llm_model            # Defines the LLM interface
│   │       ├── retriever            # Handles document retrieval using FAISS
│   │       └── rag_pipeline        # Combines LLM with retrieval for responses
│   ├── preprocess.py           # Preprocessing scripts for raw data
├── api
│   │   └── app.py                  # FastAPI application for serving the chatbot
│   └── utils
│       ├── config.py               # Centralized configuration
│       └── logger.py               # Logging utilities
├── docs
│   └── README.md                   # Project documentation
├── tests
│   └── test_pipeline.py            # Unit tests for RAG pipeline
├── artifacts
│   ├── vector_store                # FAISS index storage
│   ├── models                      # LLM and retriever models
│   └── logs                        # Application logs
├── requirements.txt                # List of project dependencies
├── .env                            # Environment variables
├── .gitignore                      # Files and directories to ignore in Git
└── README.md                       # Main documentation
```

---

## RAG Pipeline Workflow
1. **Document Preprocessing**: Input documents are cleaned and preprocessed using `data_preprocess.py`.
2. **Embedding Generation**: Text is vectorized using `sentence-transformers`. The embeddings represent the dense vector space of the documents.
3. **Vector Database**: The FAISS library is used to store and manage these vectors, enabling fast retrieval.
4. **Query Handling**: User input queries are transformed into embeddings.
5. **Document Retrieval**: Relevant documents are retrieved from the vector database based on similarity scores.
6. **Answer Generation**: The LLM generates a response by conditioning on both the retrieved documents and the user query.

---

## How to Run the Project
### Prerequisites
- Python >= 3.9
- GPU setup (optional but recommended for faster processing)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in the `.env` file:
   ```env
   VECTOR_DB_PATH=artifacts/vector_store
   MODEL_PATH=artifacts/models
   API_PORT=8000
   ```
5. Build and run the FastAPI application:
   ```bash
   uvicorn src.api.app:app --reload
   ```

### Example Usage
- Access the API at `http://127.0.0.1:8000/docs` to interact with the chatbot using Swagger UI.

---

## Future Enhancements
- Support for additional vector databases like Pinecone or Weaviate.
- Expand LLM options to include open-source alternatives like Falcon or Dolly.
- Integration with cloud services for scaling (e.g., AWS S3, Lambda).
- Add caching for frequently asked questions.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for pre-trained models and tokenizers.
- [Facebook AI](https://ai.facebook.com/tools/faiss/) for FAISS library.
- [FastAPI](https://fastapi.tiangolo.com/) for building the RESTful API.
