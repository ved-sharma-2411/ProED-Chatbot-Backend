# Hosting Backend on AWS EC2

## Step 1: Launch an EC2 Instance

1. Log in to your AWS Management Console.
2. Navigate to **EC2 Dashboard**.
3. Click **Launch Instance**.
4. Choose an Amazon Machine Image (AMI) (e.g., Ubuntu 22.04).
5. Select an instance type (e.g., t2.micro for free tier).
6. Configure instance details and add storage 30 Gb for now.
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

## Step 6: Configure .env

vim .env


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

steps to run 
-  python3 -m venv .venv
- uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1

4. Detach from the session by pressing `Ctrl+B`, then `D`.


then try 

- curl localhost:8000
you will see server is running 

then globally try 
< your public ip >:8000