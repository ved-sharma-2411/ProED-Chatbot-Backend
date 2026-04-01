# ProEd RAG API — EC2 + Nginx Deployment Guide

FastAPI server (`api_server.py`) running behind Nginx on an Amazon EC2 Ubuntu instance.

---

## Folder Structure

```
EC2-NGINX/
├── app/
│   ├── api_server.py          # FastAPI application
│   ├── rag_pinecone.py        # RAG + Pinecone pipeline
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment variable template
│   └── data/
│       ├── rag_llm_state.json # LLM rate-limit state (pre-seeded)
│       └── bm25_cache/        # BM25 index cache (auto-populated)
├── nginx/
│   └── proed-rag.conf         # Nginx reverse-proxy config
├── systemd/
│   └── proed-rag.service      # Systemd service unit
├── deploy.sh                  # Automated setup script
└── README.md                  # This guide
```

---

## Prerequisites

| What           | Requirement                                        |
| -------------- | -------------------------------------------------- |
| EC2 Instance   | Ubuntu 22.04 LTS (t3.medium or larger recommended) |
| Security Group | Inbound: port **80** (HTTP), port **22** (SSH)     |
| Keys           | Your `.pem` SSH key pair                           |
| API Keys       | `PINECONE_API_KEY` and `GROQ_API_KEY` ready        |

> For HTTPS (recommended for production) also open port **443** and follow Step 9.

---

## Step-by-Step Deployment

### Step 1 — Launch an EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Choose **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
3. Select instance type: **t3.medium** (2 vCPU, 4 GB RAM) minimum
   - Use **t3.large** if you enable `API_WARMUP_ON_START=1` (loads the sentence-transformer model into RAM)
4. Under **Key pair** — create or select an existing `.pem` key
5. Under **Network settings → Security Group**, add these inbound rules:
   - SSH: port `22`, Source: `My IP`
   - HTTP: port `80`, Source: `0.0.0.0/0`
6. Under **Storage** — set at least **20 GB** gp3 (sentence-transformers model is ~90 MB, but pip cache grows)
7. Click **Launch Instance**

---

### Step 2 — Connect to Your EC2 Instance

```bash
# On your local machine (Windows: use Git Bash or WSL)
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

Replace `<EC2_PUBLIC_IP>` with the **Public IPv4 address** shown in the EC2 console.

---

### Step 3 — Upload the EC2-NGINX Folder to EC2

From your **local machine** (open a second terminal):

```bash
# Upload the entire EC2-NGINX folder
scp -i your-key.pem -r /path/to/EC2-NGINX ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/

# Example (Windows Git Bash path):
scp -i your-key.pem -r "C:/Users/Lenovo/Desktop/web-dev/scrapping/ProED-Chatbot-Backend/EC2-NGINX" ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/
```

---

### Step 4 — Run the Deploy Script

Back in your **SSH session**:

```bash
cd /home/ubuntu/EC2-NGINX
chmod +x deploy.sh
bash deploy.sh
```

The script will:

- Install Python 3, pip, venv, Nginx
- Create `/home/ubuntu/proed-rag/app/` with all app files
- Create a Python virtual environment at `/home/ubuntu/proed-rag/venv/`
- Install all Python dependencies
- Configure Nginx and enable it
- Register the systemd service (but NOT start it yet)

> **Manual setup (if deploy.sh fails)**
>
> Run these commands on the EC2 instance:
>
> ```bash
> # Update packages
> sudo apt-get update -y
>
> # Install Python + venv + Nginx
> sudo apt-get install -y python3 python3-pip python3-venv nginx curl
>
> # Create app directory
> sudo mkdir -p /home/ubuntu/proed-rag/app
> sudo chown -R ubuntu:ubuntu /home/ubuntu/proed-rag
>
> # Copy app files (assumes EC2-NGINX is in /home/ubuntu)
> cp -r /home/ubuntu/ProED-Chatbot-Backend/EC2-NGINX/app/* /home/ubuntu/proed-rag/app/
>
> # Create venv + install deps
> python3 -m venv /home/ubuntu/proed-rag/venv
> source /home/ubuntu/proed-rag/venv/bin/activate
> pip install --upgrade pip
> pip install -r /home/ubuntu/proed-rag/app/requirements.txt
>
> # Configure Nginx
> sudo cp /home/ubuntu/ProED-Chatbot-Backend/EC2-NGINX/nginx/proed-rag.conf /etc/nginx/sites-available/proed-rag
> sudo ln -sf /etc/nginx/sites-available/proed-rag /etc/nginx/sites-enabled/proed-rag
> sudo rm -f /etc/nginx/sites-enabled/default
> sudo nginx -t
> sudo systemctl reload nginx
> sudo systemctl enable nginx
>
> # Configure systemd service
> sudo cp /home/ubuntu/ProED-Chatbot-Backend/EC2-NGINX/systemd/proed-rag.service /etc/systemd/system/proed-rag.service
> sudo systemctl daemon-reload
> sudo systemctl enable proed-rag
> ```

---

### Step 5 — Configure Your Environment Variables

```bash
cp /home/ubuntu/proed-rag/app/.env.example /home/ubuntu/proed-rag/app/.env
nano /home/ubuntu/proed-rag/app/.env
```

Fill in at minimum:

```env
PINECONE_API_KEY=your_actual_pinecone_key
GROQ_API_KEY=your_actual_groq_key
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`.

---

### Step 6 — Start the FastAPI Service

```bash
sudo systemctl start proed-rag
sudo systemctl status proed-rag
```

You should see `Active: active (running)`. Check logs if it fails:

```bash
sudo journalctl -u proed-rag -n 50 --no-pager
# or
tail -f /var/log/proed-rag/error.log
```

---

### Step 7 — Verify Everything is Working

```bash
# Test locally on the server (bypasses Nginx)
curl http://127.0.0.1:8000/health

# Test through Nginx (public-facing)
curl http://<EC2_PUBLIC_IP>/health

# Expected response:
# {"status":"ok"}
```

Open in browser: `http://<EC2_PUBLIC_IP>/docs` — you'll see the interactive FastAPI Swagger UI.

---

### Step 8 — Test the API Endpoints

**Root:**

```bash
curl http://<EC2_PUBLIC_IP>/
```

**Ask endpoint:**

```bash
curl -X POST http://<EC2_PUBLIC_IP>/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is student eligibility?", "top_k": 5, "namespace": "default"}'
```

**Query-only endpoint:**

```bash
curl -X POST http://<EC2_PUBLIC_IP>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is student eligibility?", "top_k": 5}'
```

---

### Step 9 — (Optional) Add HTTPS with Let's Encrypt

If you have a domain name pointed to your EC2 IP:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

Certbot will automatically edit your Nginx config and set up auto-renewal.

---

## Managing the Service

| Action           | Command                                 |
| ---------------- | --------------------------------------- |
| Start            | `sudo systemctl start proed-rag`        |
| Stop             | `sudo systemctl stop proed-rag`         |
| Restart          | `sudo systemctl restart proed-rag`      |
| Status           | `sudo systemctl status proed-rag`       |
| View logs (live) | `sudo journalctl -u proed-rag -f`       |
| View access logs | `tail -f /var/log/proed-rag/access.log` |
| View error logs  | `tail -f /var/log/proed-rag/error.log`  |

The service is set to **auto-start on reboot** (`systemctl enable`).

---

## Updating the App Code

When you change `api_server.py` or `rag_pinecone.py` locally (inside `ProED-Chatbot-Backend/EC2-NGINX/app`):

```bash
# 1. Upload new files from local machine
scp -i your-key.pem ProED-Chatbot-Backend/EC2-NGINX/app/api_server.py ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/proed-rag/app/
scp -i your-key.pem ProED-Chatbot-Backend/EC2-NGINX/app/rag_pinecone.py ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/proed-rag/app/

# 2. Restart the service on EC2
sudo systemctl restart proed-rag
```

---

## Troubleshooting

**Service fails to start:**

```bash
sudo journalctl -u proed-rag -n 100 --no-pager
```

Most common cause: missing `.env` file or wrong API keys.

**502 Bad Gateway from Nginx:**
The FastAPI service is not running. Run `sudo systemctl start proed-rag` and check logs.

**Port 80 not accessible:**
Check EC2 Security Group — ensure inbound rule for port 80 from `0.0.0.0/0` exists.

**Model download taking long on first start:**
`sentence-transformers/all-MiniLM-L6-v2` downloads ~90 MB on first run. Wait 1-2 minutes. Use `journalctl -u proed-rag -f` to watch progress.

**Out of memory (OOM):**
Upgrade to a larger instance (`t3.large` or `t3.xlarge`). The sentence-transformer model uses ~300 MB RAM.

---

## Architecture

```
Internet
   │
   ▼ port 80
 Nginx  (reverse proxy, /etc/nginx/sites-available/proed-rag)
   │
   ▼ localhost:8000
 Gunicorn + UvicornWorker  (systemd: proed-rag.service)
   │
   ▼
 FastAPI app  (api_server.py)
   │
   ├──► Pinecone  (vector search)
   └──► Groq API  (LLM via OpenAI-compatible client)
```
