# ⬡ NeuralMesh — Decentralised Idle Compute Platform

> **Hackathon prototype** — pure Python, zero cloud, zero blockchain.

---

## Quick start

```bash
pip install -r requirements.txt
bash run_demo.sh          # starts coordinator + 2 workers + Streamlit dashboard
```

Open **http://localhost:8501** for the live dashboard.  
Open **http://localhost:8000/docs** for the coordinator's interactive API docs.

---

## Library ↔ feature mapping

| Library | Used for |
|---|---|
| `fastapi` + `uvicorn` | Coordinator REST API · Worker REST API |
| `pydantic` | All request/response models (NodeSpec, TaskSpec, …) |
| `pulp` | Integer linear programming scheduler (DRF fair-share) |
| `networkx` | Mesh graph — topology, density, connectivity checks |
| `numpy` | ANN / CNN training · GPoW gradient computation · Monte Carlo |
| `psutil` | Worker heartbeat — CPU%, memory%, core count |
| `cryptography` | HMAC-SHA256 task signing · GPoW challenge nonces |
| `requests` | Worker→coordinator HTTP calls (heartbeat, result, checkpoint) |
| `streamlit` | Real-time dashboard — all 6 tabs |
| `pandas` | DataFrames for all dashboard tables |
| `matplotlib` | Charts — usage history, heatmap, leaderboard, mesh topology |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Internet / LAN mesh                                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Coordinator  (FastAPI · :8000)               │   │
│  │  ┌──────────┐ ┌─────────┐ ┌──────────────┐   │   │
│  │  │ Scheduler│ │  GPoW   │ │ Credit ledger│   │   │
│  │  │  (PuLP)  │ │ engine  │ │ (streaks,    │   │   │
│  │  │   DRF    │ │(crypto) │ │  pledges,    │   │   │
│  │  └──────────┘ └─────────┘ │  achievements│   │   │
│  │  ┌──────────┐ ┌─────────┐ └──────────────┘   │   │
│  │  │ Node     │ │ NetworkX│                     │   │
│  │  │ registry │ │  mesh   │                     │   │
│  │  └──────────┘ └─────────┘                     │   │
│  └──────────────────────────────────────────────┘   │
│          ↑heartbeat / ↓dispatch                      │
│   ┌──────────────┐    ┌──────────────┐               │
│   │ Worker α     │    │ Worker β     │  … N workers  │
│   │ FastAPI :8001│    │ FastAPI :8002│               │
│   │ psutil       │    │ psutil       │               │
│   │ numpy executor│   │ numpy executor│              │
│   └──────────────┘    └──────────────┘               │
│                                                      │
│   ┌────────────────────────────────────────────┐    │
│   │  Streamlit Dashboard  (:8501)               │    │
│   │  Overview · Nodes · Jobs · Credits · Mesh  │    │
│   └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## Brownie-point features

### 1 · Incentive engine
- **Streak multiplier** — consecutive days of contribution scale credit earnings from 1.0× up to **2.0×** (30-day "Legend" tier).
- **Compute futures** — contributors pledge idle windows in advance; honouring the pledge earns a **+20% bonus**; cancelling within 1 hour incurs a −50 ⬡ penalty.
- **Achievements** — 10 milestones (first job, week warrior, GPU guru, legend, …) visible in the dashboard and credits tab.
- **Dominant Resource Fairness (DRF)** in the PuLP scheduler — nodes with high accumulated balances are slightly deprioritised so new contributors always get work.

### 2 · Fault tolerance
- **Heartbeat monitor** (`threading`) — detects node timeout in ≤30 s, marks node `failed`.
- **Checkpoint-based recovery** — workers call `POST /checkpoint` every 2 epochs with a SHA-256 of the weight bytes.  On node failure the coordinator replays from the saved epoch, not from epoch 0.
- **Three recovery modes**
  | Mode | Description | Credit cost |
  |---|---|---|
  | Checkpoint replay | Default — resume from last checkpoint | 1× |
  | Hot standby | Shadow receives every gradient update live; zero replay | 2× |
  | Speculative rollback | GPoW mismatch triggers automatic rollback to last verified checkpoint | 1× |
- **GPoW adversarial recovery** — if a shadow node's gradient hash diverges, the suspicious node is slashed (−30 ⬡, −15 trust) and the job is rescheduled to clean nodes.

### 3 · Usage dashboard (Streamlit)
| Tab | Content |
|---|---|
| Overview | Live stats metrics, rolling active-nodes & running-jobs chart, activity feed |
| Nodes | Registry table, trust-score histogram, status pie chart, per-node resource bars |
| Jobs | Job table, status pie, checkpoint progress bars, fault-recovery log |
| Credits & Incentives | Personal ledger, streak info, leaderboard bar chart, contribution heatmap, multiplier tier table |
| Mesh topology | NetworkX graph rendered with matplotlib — node colours by type, density & connectivity |
| Submit job | Full job submission form with cost estimator |

---

## Running multiple workers (distributed)

```bash
# Machine A (coordinator)
cd coordinator
uvicorn main:app --host 0.0.0.0 --port 8000

# Machine B (GPU worker)
cd worker
python main.py --name gpu-node-1 --host <machine-B-ip> --port 8001 \
               --coordinator http://<machine-A-ip>:8000 \
               --node-type GPU --vram 8.0

# Machine C (CPU worker)
cd worker
python main.py --name cpu-node-1 --host <machine-C-ip> --port 8001 \
               --coordinator http://<machine-A-ip>:8000

# Dashboard (anywhere with network access)
streamlit run dashboard/app.py
# Enter coordinator URL in the sidebar
```

---

## GPoW verification — how it works

```
Coordinator
  │
  ├─ generate verification_seed  (random int)
  │
  ├─ send TaskSpec + seed ──────────────────────► Primary worker
  │                                                  │
  ├─ send TaskSpec + seed (is_shadow=True) ────► Shadow worker(s)
  │                                                  │
  │                      Both workers:               │
  │            ┌──────────────────────────┐          │
  │            │ 1. Init model (seed=0)   │          │
  │            │ 2. Draw mini-batch using │          │
  │            │    verification_seed     │          │
  │            │ 3. Forward pass          │          │
  │            │ 4. Backward (lr=0)       │          │
  │            │ 5. hash(gradient bytes)  │          │
  │            └──────────────────────────┘          │
  │                                                  │
  ◄─ POST /result {gradient_hash, loss, accuracy} ───┘
  │
  └─ Compare hashes:
       match   → verified → reward primary + shadows
       mismatch → slash suspicious node → reschedule
```

Because numpy training is fully deterministic given the same seeds, honest nodes always produce the same gradient hash on the verification batch regardless of their local hardware.
