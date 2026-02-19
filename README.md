# CoFina RAG Agent

A simple RAG (Retrieval-Augmented Generation) agent using LangChain, ChromaDB, and CMU's AI Gateway.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

**Requires python 3.12 or older, if you face issues with chromaDB or pydantic use 3.12 or older, this requires you to have installed that runtime on your system and then to create a virtual environment with that specific runtime**

```bash
py -3.12 -m venv venv
venv/Scripts/activate
```
**Remember to create venv in root**


2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key:**
   - Copy `.env.example` to `.env`
   - Add your CMU API key to `.env` as `OPENAI_API_KEY`

4. **Add PDF documents:**
   - Place PDF files in `src/docs/` directory

## Usage

From the root directory, run the following CLI command:
```bash
python setupDB.py
```

Run the CLI:
```bash
cd src
python app.py
```

The app automatically caches the vector store and only re-processes documents when they change (based on filename and modification time).

To force reindexing (e.g., after adding/removing PDFs):
```bash
python app.py --reindex
```

## Architecture

- **chunking.py**: Semantic chunking with OpenAI embeddings (via CMU gateway)
- **index.py**: ChromaDB vector store management
- **retriever.py**: RAG chain with Gemini 1.5 Pro (via CMU gateway)
- **app.py**: CLI interface

## Free Embedding Alternatives

If you want to avoid Gemini embedding costs:
- `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- `BAAI/bge-small-en-v1.5` (local, free)

To use local embeddings, replace in `chunking.py` and `index.py`:
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
