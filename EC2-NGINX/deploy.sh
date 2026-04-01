#!/bin/bash
# =============================================================================
# deploy.sh — One-shot setup script for ProEd RAG on Ubuntu EC2
# Run as: bash deploy.sh
# =============================================================================
set -euo pipefail

APP_DIR="/home/ubuntu/proed-rag"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/proed-rag"
SERVICE_NAME="proed-rag"

echo "==> [1/7] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv nginx curl

echo "==> [2/7] Creating app directory..."
sudo mkdir -p "$APP_DIR"
sudo chown ubuntu:ubuntu "$APP_DIR"
mkdir -p "$APP_DIR/app/data/bm25_cache"

echo "==> [3/7] Copying app files..."
cp -r ./app/* "$APP_DIR/app/"

echo "==> [4/7] Creating Python virtual environment and installing deps..."
python3 -m venv "$VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
	echo "ERROR: venv was not created at $VENV_DIR"
	echo "Make sure python3-venv is installed and you are not running from a read-only path."
	exit 1
fi
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_DIR/app/requirements.txt"

echo "==> [5/7] Configuring Nginx..."
sudo cp ./nginx/proed-rag.conf /etc/nginx/sites-available/proed-rag
sudo ln -sf /etc/nginx/sites-available/proed-rag /etc/nginx/sites-enabled/proed-rag
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl enable nginx

echo "==> [6/7] Setting up systemd service..."
sudo mkdir -p "$LOG_DIR"
sudo chown ubuntu:ubuntu "$LOG_DIR"
sudo cp ./systemd/proed-rag.service /etc/systemd/system/proed-rag.service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "==> [7/7] IMPORTANT: Set your .env file before starting the service"
echo "    cp $APP_DIR/app/.env.example $APP_DIR/app/.env"
echo "    nano $APP_DIR/app/.env   # fill in PINECONE_API_KEY and GROQ_API_KEY"
echo ""
echo "    Then start the service:"
echo "    sudo systemctl start proed-rag"
echo "    sudo systemctl status proed-rag"
echo ""
echo "Deploy script complete."
