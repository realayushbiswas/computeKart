"""
NeuralMesh Worker Node
======================
Runs as an independent FastAPI process.  Multiple workers can run on
different ports on the same machine or across the network.

Responsibilities
----------------
* Register with the coordinator on startup
* Send psutil-based heartbeats every 10 seconds (earns credits for idle time)
* Accept /execute requests from the coordinator
* Run training/simulation tasks via TaskExecutor (numpy)
* Send gradient results + checkpoints back to the coordinator
* Gracefully handle mid-job interruptions (TaskExecutor is threaded)

Usage
-----
    # Worker 1 (default)
    python main.py --name worker-alpha --port 8001

    # Worker 2
    python main.py --name worker-beta --port 8002 --coordinator http://localhost:8000
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import psutil
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add parent dir so executor can be imported cleanly
sys.path.insert(0, os.path.dirname(__file__))
from executor import TaskExecutor

# ---------------------------------------------------------------------------
# CLI args — parsed before uvicorn rewrites sys.argv
_parser = argparse.ArgumentParser(description="NeuralMesh Worker Node")
_parser.add_argument("--name",        default=f"worker-{os.getpid()}")
_parser.add_argument("--host",        default="127.0.0.1")
_parser.add_argument("--port",        type=int, default=8001)
_parser.add_argument("--coordinator", default=os.environ.get("COORDINATOR_URL", "http://127.0.0.1:8000"))
_parser.add_argument("--node-type",   default="CPU", choices=["CPU", "GPU", "Browser"])
_parser.add_argument("--vram",        type=float, default=0.0)
_args, _extra = _parser.parse_known_args()

COORDINATOR_URL = _args.coordinator
NODE_STATE: dict = {"node_id": None}
EXECUTOR = TaskExecutor(coordinator_url=COORDINATOR_URL, node_state=NODE_STATE)

# ---------------------------------------------------------------------------
app = FastAPI(title=f"NeuralMesh Worker — {_args.name}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
class ExecuteRequest(BaseModel):
    job_id:            str
    task_type:         str  = "train_ann"
    epochs:            int  = 5
    batch_size:        int  = 32
    start_epoch:       int  = 0
    input_dim:         int  = 64
    hidden_dim:        int  = 64
    output_dim:        int  = 10
    dataset_size:      int  = 512
    verification_seed: Optional[int] = None
    is_shadow:         bool = False
    min_trust:         float = 40.0
    redundancy:        int  = 2
    checkpoint_epoch:  int  = 0


# ---------------------------------------------------------------------------
@app.post("/execute")
def execute(req: ExecuteRequest):
    """Accept a job and run it in a background thread."""
    threading.Thread(target=_run_job, args=(req,), daemon=True).start()
    return {"status": "accepted", "job_id": req.job_id}


def _run_job(req: ExecuteRequest):
    try:
        result = EXECUTOR.run(req)
        node_id = NODE_STATE.get("node_id") or "unknown"
        requests.post(
            f"{COORDINATOR_URL}/result",
            json={
                "job_id":         req.job_id,
                "node_id":        node_id,
                "epoch":          req.epochs,
                "gradient_hash":  result["gradient_hash"],
                "loss":           result["final_loss"],
                "accuracy":       result["final_accuracy"],
                "is_shadow":      req.is_shadow,
            },
            timeout=15,
        )
    except Exception as exc:
        print(f"[{_args.name}] Job {req.job_id} error: {exc}")


# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "node_id":     NODE_STATE.get("node_id"),
        "name":        _args.name,
        "node_type":   _args.node_type,
        "coordinator": COORDINATOR_URL,
        "cpu_percent": psutil.cpu_percent(),
        "mem_percent": psutil.virtual_memory().percent,
        "running_jobs": len(EXECUTOR.running_jobs),
        "timestamp":   datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Background: register + heartbeat

def _register():
    payload = {
        "name":       _args.name,
        "host":       _args.host,
        "port":       _args.port,
        "node_type":  _args.node_type,
        "vram_gb":    _args.vram,
        "cpu_cores":  psutil.cpu_count(logical=True),
        "trust_score": 70.0,
    }
    for attempt in range(10):
        try:
            r = requests.post(f"{COORDINATOR_URL}/register", json=payload, timeout=10)
            if r.ok:
                data = r.json()
                NODE_STATE["node_id"] = data["node_id"]
                EXECUTOR._state = NODE_STATE
                print(f"[{_args.name}] Registered — node_id={NODE_STATE['node_id']} credits={data['initial_credits']}")
                return
        except Exception as exc:
            print(f"[{_args.name}] Registration attempt {attempt+1} failed: {exc}")
        time.sleep(3)
    print(f"[{_args.name}] Could not register with coordinator after 10 attempts.")


def _heartbeat_loop():
    """Send psutil metrics to coordinator every 10 seconds."""
    while True:
        time.sleep(10)
        node_id = NODE_STATE.get("node_id")
        if not node_id:
            continue
        try:
            cpu  = psutil.cpu_percent(interval=1)
            mem  = psutil.virtual_memory().percent
            payload = {
                "node_id":       node_id,
                "cpu_percent":   cpu,
                "memory_percent": mem,
                "active_jobs":   len(EXECUTOR.running_jobs),
            }
            r = requests.post(f"{COORDINATOR_URL}/heartbeat", json=payload, timeout=5)
            if r.ok:
                data = r.json()
                print(f"[{_args.name}] ♡ heartbeat | cpu={cpu:.0f}% | balance={data.get('balance', '?')} ⬡ | streak={data.get('streak', 0)}d ×{data.get('multiplier', 1)}")
        except Exception as exc:
            print(f"[{_args.name}] Heartbeat failed: {exc}")


@app.on_event("startup")
def startup():
    threading.Thread(target=_register,       daemon=True).start()
    threading.Thread(target=_heartbeat_loop, daemon=True).start()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting worker '{_args.name}' on {_args.host}:{_args.port}")
    print(f"Coordinator: {COORDINATOR_URL}")
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="warning")
