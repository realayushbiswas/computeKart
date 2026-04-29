#!/usr/bin/env bash
# NeuralMesh demo launcher
# Usage: bash run_demo.sh
# Starts: coordinator (8000), worker-alpha (8001), worker-beta (8002), dashboard (8501)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  ⬡  NeuralMesh — Demo Launcher"
echo "=========================================="

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r "$ROOT/requirements.txt" -q

# Start coordinator
echo "[2/4] Starting coordinator on :8000..."
cd "$ROOT/coordinator"
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning &
COORD_PID=$!
sleep 2

# Start two workers
echo "[3/4] Starting worker-alpha on :8001 and worker-beta on :8002..."
cd "$ROOT/worker"
python main.py --name worker-alpha --port 8001 --node-type CPU &
W1_PID=$!
sleep 1
python main.py --name worker-beta  --port 8002 --node-type CPU &
W2_PID=$!
sleep 2

# Fund demo user
echo "      Funding demo user 'demo-user' with 500 credits via coordinator..."
curl -s -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"name":"demo-user","host":"localhost","port":9999,"node_type":"CPU","trust_score":80}' \
  > /dev/null

# Start Streamlit dashboard
echo "[4/4] Starting dashboard on :8501..."
cd "$ROOT"
streamlit run dashboard/app.py --server.port 8501 --server.headless true &
DASH_PID=$!

echo ""
echo "  Coordinator : http://localhost:8000/docs"
echo "  Dashboard   : http://localhost:8501"
echo "  Worker-alpha: http://localhost:8001/health"
echo "  Worker-beta : http://localhost:8002/health"
echo ""
echo "  Press Ctrl+C to stop all services."

trap "kill $COORD_PID $W1_PID $W2_PID $DASH_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait




