# Scrapping Pipeline (HTML → Chunks → Embeddings → Pinecone RAG)

## What each script does

- **html-scrap.py**  
  Fetches raw ECFR HTML and saves it to `data/raw_html/ecfr/section_668_32.html`.

- **html-parse.py**  
  Parses saved HTML, cleans text, detects section/subsections, and outputs:
  - `section_668_32.json`
  - `section_668_32.txt`

- **html-chunk.py**  
  Builds logical chunks from parsed JSON:
  - hierarchy-aware subsection chunking
  - token-based size control (min/max)
  - overlap between chunks
  - chunk metadata

- **html-embeddings.py**  
  Legacy OpenAI embeddings script (not used in the Groq-only flow).

- **html-embeddings-st.py**  
  Creates local embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).

- **rag_pinecone.py**  
  End-to-end Pinecone RAG script:
  - ingestion (embed + upsert)
  - semantic retrieval
  - final answer generation with Groq chat + rate-limit safeguards

---

## Quick flow

1. `python html-scrap.py`
2. `python html-parse.py`
3. `python html-chunk.py`
4. Embeddings:

- `python html-embeddings-st.py`

5. Pinecone ingest/query:
   - `python rag_pinecone.py ingest --input data/raw_html/ecfr/section_668_32_chunks.json`
   - `python rag_pinecone.py query --question "student eligibility under 34 CFR 668.32"`
   - `python rag_pinecone.py ask --question "..."`

## Required env vars

- `PINECONE_API_KEY`
- `GROQ_API_KEY` (for chat and optional API embeddings)