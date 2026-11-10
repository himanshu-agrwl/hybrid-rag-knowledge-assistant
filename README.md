# Hybrid RAG-Based Knowledge Assistant

A modular Retrieval-Augmented Generation (RAG) system that enables users to ask questions about documents and receive context-grounded answers using a Large Language Model (LLM).

The project combines semantic search and keyword-based retrieval to find relevant document chunks before passing them to Gemini for answer generation.

**Status:** Work in Progress

## Features

- Recursive document ingestion for TXT and Markdown files.
- Paragraph-based document chunking.
- Semantic search using Sentence Transformers.
- Keyword-based retrieval using BM25.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Context-grounded answer generation using Gemini.
- Evidence synthesis across multiple retrieved chunks.
- Abstention when the available context is insufficient.
- Initial retrieval and generation evaluation pipelines.

## Architecture

```text
Documents (TXT / MD)
        |
        v
Document Ingestion
        |
        v
Document Chunking
        |
        v
Embedding Generation
        |
        +----------------------+
        |                      |
        v                      v
   Semantic Search         BM25 Search
        |                      |
        +----------+-----------+
                   |
                   v
          Reciprocal Rank Fusion
                   |
                   v
          Relevant Context
                   |
                   v
             Gemini LLM
                   |
                   v
          Grounded Answer
                   |
                   v
          Answer + Sources
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Keyword Retrieval | BM25 |
| Hybrid Ranking | Reciprocal Rank Fusion |
| LLM | Gemini 3.5 Flash Lite |
| Evaluation | Recall@K, Precision@K, MRR |

## Project Structure

```text
RAG-knowledge-assistant/
│
├── data/
│   ├── documents/
│   ├── evaluation_queries.json
│   └── evaluation_generation.json
│
├── src/
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── keyword_search.py
│   ├── bm25_search.py
│   ├── hybrid_search.py
│   ├── generator.py
│   └── evaluation.py
│
├── main.py
├── evaluate_retrieval.py
├── evaluate_generation.py
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/himanshu-agrwl/hybrid-rag-knowledge-assistant.git
cd hybrid-rag-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API

Create a `.env` file in the project root using `.env.example` as a template.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Never commit your `.env` file.

### 5. Run the application

```bash
python main.py
```

### 6. Run retrieval evaluation

```bash
python evaluate_retrieval.py
```

### 7. Run generation evaluation

```bash
python evaluate_generation.py
```

## Evaluation

The project includes an initial retrieval evaluation dataset and a separate generation evaluation dataset.

Retrieval evaluation uses:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)

Generation evaluation currently tests:

- Answerable and unanswerable questions.
- Evidence-grounded responses.
- Multi-chunk evidence synthesis.
- Insufficient-context handling.

Evaluation is ongoing. Current results are preliminary and based on a small document collection and test dataset.

## Current Limitations

- Supports TXT and Markdown documents only.
- Uses an in-memory retrieval index.
- No persistent vector database integration yet.
- No cross-encoder reranking yet.
- No web interface or deployment.
- Generation and faithfulness evaluation are still being developed.

## Future Improvements

- Persistent vector storage using Qdrant.
- Cross-encoder reranking.
- Advanced retrieval strategies.
- PDF and DOCX document support.
- Local LLM integration and comparison with API-based models.
- FastAPI backend and Streamlit interface.
- Improved evaluation and deployment.

## Author

Developed as an ongoing AI/ML engineering project focused on information retrieval, LLM integration, and Retrieval-Augmented Generation.