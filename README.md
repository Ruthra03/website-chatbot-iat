# Website chatbot (IAT Networks)

A retrieval-augmented chatbot that crawls a configured website, embeds the text into a FAISS vector store, and answers questions with [Groq](https://groq.com/) (Llama 3.3) via LangChain. The app includes a Streamlit UI and a terminal mode.

## Requirements

- Python 3.9 or newer (recommended)
- A [Groq API key](https://console.groq.com/)

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   cd website-chatbot
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure the API key. The code reads **`GROK_API_KEY`** from the environment (see `chatbot.py`). For local development, create a `.env` file in this directory:

   ```
   GROK_API_KEY=your_key_here
   ```

   `.env` is listed in `.gitignore` and should not be committed.

3. **Streamlit only:** If you run without a `.env` file, set the same key under `GROK_API_KEY` in [Streamlit secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management) (for example `.streamlit/secrets.toml` locally).

## Run the web UI

```bash
streamlit run app.py
```

The first startup may take a while: the crawler may run and embeddings are built. Later runs reuse the saved FAISS index under `faiss_index/` when present.

## Run the terminal chatbot

```bash
python chatbot.py
```

Force a full re-crawl and rebuild of the vector store:

```bash
python chatbot.py --rebuild
```

## Configuration

Editable in `chatbot.py`:

| Setting | Purpose |
|--------|---------|
| `BASE_URL` | Root URL to crawl (same site only) |
| `MAX_PAGES` | Upper bound on pages to visit |
| `MODEL` | Groq model name |
| `VECTOR_STORE_PATH` | Directory for the persisted FAISS index |

## Project layout

| File / folder | Role |
|---------------|------|
| `app.py` | Streamlit chat interface |
| `chatbot.py` | Crawler, embeddings, RAG, `Chatbot` class |
| `faiss_index/` | Saved vector store (generated; safe to delete to rebuild) |
| `requirements.txt` | Python dependencies |
