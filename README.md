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

---

# Hosting Backend on AWS EC2

## Step 1: Launch an EC2 Instance

1. Log in to your AWS Management Console.
2. Navigate to **EC2 Dashboard**.
3. Click **Launch Instance**.
4. Choose an Amazon Machine Image (AMI) (e.g., Ubuntu 22.04).
5. Select an instance type (e.g., t2.micro for free tier).
6. Configure instance details and add storage as needed.
7. Add a security group allowing **HTTP (port 80)** and **SSH (port 22)**.
8. Launch the instance and download the private key (.pem file).

## Step 2: Connect to the EC2 Instance

1. Open a terminal on your local machine.
2. Run the following command to connect via SSH:
   ```bash
   ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
   ```

## Step 3: Install Required Software

1. Update the package list:
   ```bash
   sudo apt update
   ```
2. Install Python, pip, and Git:
   ```bash
   sudo apt install python3 python3-pip git -y
   ```

## Step 4: Clone the Repository

1. Clone your backend repository:
   ```bash
   git clone https://github.com/ved-sharma-2411/ProED-Chatbot-Backend.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ProED-Chatbot-Backend
   ```

## Step 5: Set Up Virtual Environment

1. Install `venv`:
   ```bash
   sudo apt install python3-venv -y
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements_pipeline.txt
   ```

## Step 6: Run the Backend

1. Start the backend server:
   ```bash
   python run_full_pipeline_api.py
   ```
2. Ensure the server is running by visiting `http://<EC2_PUBLIC_IP>:8000`.

## Step 7: Configure Firewall (Optional)

1. Allow traffic on port 8000:
   ```bash
   sudo ufw allow 8000
   ```

## Step 8: Keep the Server Running

1. Install `tmux` to keep the server running:
   ```bash
   sudo apt install tmux -y
   ```
2. Start a new `tmux` session:
   ```bash
   tmux new -s backend
   ```
3. Run the server inside the `tmux` session.
4. Detach from the session by pressing `Ctrl+B`, then `D`.

---

Your backend is now hosted on AWS EC2.

ssh -i "vedant-test-server.pem" ubuntu@ec2-13-222-115-184.compute-1.amazonaws.com

uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
