# Setup and Installation — Lunar PSR Terminal

## Prerequisites
- macOS (Apple Silicon recommended with MPS support) or Linux with CUDA.
- Python 3.9+ (Python 3.10+ recommended).
- Node.js 18+ (Node 22+ recommended).

## Backend Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
# (Optional for report validation testing)
pip install -r backend/requirements-dev.txt

# Verify hardware & dependencies
python scripts/check_deps.py

# Start Backend Server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The application will launch at `http://localhost:5173` and automatically proxy API calls to port `8000`.
