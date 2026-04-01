"""
Start the FastAPI server locally and expose it globally via ngrok.

Usage:
    python run_with_ngrok.py

Optional env var:
    NGROK_AUTHTOKEN  – your ngrok auth token (needed for persistent URLs / longer sessions)
                       Get one free at https://dashboard.ngrok.com/authtokens
"""
import os
import threading
import time

import uvicorn
from pyngrok import conf, ngrok

# ── optional: set your ngrok auth token ──────────────────────────────────────
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "2z3Uf8L4I5NFcuqRzO7GPItBKpG_7V7H9uFjutRVPtxa52Syt")   # or paste your token here
if NGROK_AUTHTOKEN:
    conf.get_default().auth_token = NGROK_AUTHTOKEN

HOST = "127.0.0.1"
PORT = 8000


def start_server():
    uvicorn.run("api_server:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    # Start uvicorn in a background thread
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # Give the server a moment to bind
    time.sleep(2)

    # Open an ngrok tunnel to the local port
    tunnel = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url

    print("\n" + "=" * 60)
    print(f"  Local  : http://{HOST}:{PORT}")
    print(f"  Public : {public_url}")
    print(f"  Docs   : {public_url}/docs")
    print(f"  Health : {public_url}/health")
    print("=" * 60)
    print("  Press Ctrl+C to stop.\n")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down…")
        ngrok.disconnect(public_url)
        ngrok.kill()
