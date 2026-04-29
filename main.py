"""
NeuralMesh Coordinator
======================
Central orchestrator that:
  - Maintains the node registry and mesh graph (networkx)
  - Schedules jobs via DRF / ILP (PuLP)
  - Verifies results via Gradient Proof-of-Work (GPoW)
  - Manages the credit ledger (streaks, pledges, achievements)
  - Handles fault detection and checkpoint-based recovery
  - Exposes a REST API consumed by workers and the Streamlit dashboard
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ledger import CreditLedger
from models import (
    CheckpointData,
    GradientResult,
    HeartbeatRequest,
    JobStatus,
    NodeSpec,
    NodeStatus,
    TaskSpec,
)
from scheduler import fair_schedule
from trust import GPoWEngine

# ---------------------------------------------------------------------------
app = FastAPI(title="NeuralMesh Coordinator", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# In-memory state
nodes:            Dict[str, NodeSpec]           = {}
jobs:             Dict[str, TaskSpec]           = {}
checkpoints:      Dict[str, CheckpointData]     = {}   # job_id -> latest
gradient_results: Dict[str, List[GradientResult]] = {}
activity_log:     List[Dict]                    = []
resource_history: List[Dict]                    = []   # rolling 60-point buffer

mesh   = nx.Graph()
ledger = CreditLedger()
gpow   = GPoWEngine()

HEARTBEAT_TIMEOUT  = 30   # seconds before a node is marked failed
CHECKPOINT_INTERVAL = 2   # epochs between checkpoints

# ---------------------------------------------------------------------------
# Helpers

def _log(event_type: str, message: str, node_id: str = None, job_id: str = None):
    activity_log.insert(0, {
        "timestamp": datetime.utcnow().isoformat(),
        "type":      event_type,
        "message":   message,
        "node_id":   node_id,
        "job_id":    job_id,
    })
    del activity_log[200:]   # keep last 200 events


def _estimate_cost(job: TaskSpec) -> float:
    base = {"train_ann": 20, "train_cnn": 40, "simulate": 15, "preprocess": 10}.get(job.task_type, 20)
    return round(base * job.epochs * job.redundancy * 0.5, 2)


# ---------------------------------------------------------------------------
# Background: heartbeat monitor

def _heartbeat_monitor():
    while True:
        time.sleep(5)
        now = datetime.utcnow()
        for node_id, node in list(nodes.items()):
            if node.last_heartbeat and node.status == NodeStatus.ACTIVE:
                last = datetime.fromisoformat(node.last_heartbeat)
                if (now - last).total_seconds() > HEARTBEAT_TIMEOUT:
                    node.status = NodeStatus.FAILED
                    _log("fault", f"Heartbeat timeout — {node.name} marked failed", node_id=node_id)
                    threading.Thread(target=_handle_node_failure, args=(node_id,), daemon=True).start()


def _handle_node_failure(failed_node_id: str):
    for job_id, job in list(jobs.items()):
        if failed_node_id in job.assigned_nodes and job.status == JobStatus.RUNNING:
            ckpt_ep = job.checkpoint_epoch
            job.status = JobStatus.QUEUED
            job.assigned_nodes = [n for n in job.assigned_nodes if n != failed_node_id]
            _log("recovery", f"Job {job_id} rescheduling from checkpoint epoch {ckpt_ep}", job_id=job_id)
            time.sleep(2)
            _schedule_and_dispatch(job_id)


threading.Thread(target=_heartbeat_monitor, daemon=True).start()


# ---------------------------------------------------------------------------
# Background: resource-history sampler

def _resource_sampler():
    while True:
        active = [n for n in nodes.values() if n.status == NodeStatus.ACTIVE]
        snapshot = {
            "timestamp":    datetime.utcnow().isoformat(),
            "active_nodes": len(active),
            "running_jobs": sum(1 for j in jobs.values() if j.status == JobStatus.RUNNING),
        }
        resource_history.insert(0, snapshot)
        del resource_history[120:]
        time.sleep(5)


threading.Thread(target=_resource_sampler, daemon=True).start()


# ---------------------------------------------------------------------------
# Scheduling + dispatch

def _schedule_and_dispatch(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return

    available = [
        n for n in nodes.values()
        if n.status == NodeStatus.ACTIVE and n.trust_score >= job.min_trust
           and n.node_id not in job.assigned_nodes
    ]
    if not available:
        _log("warn", f"No eligible nodes for job {job_id}", job_id=job_id)
        return

    schedule = fair_schedule(job, available, ledger.get_all_balances())
    if not schedule:
        _log("warn", f"Scheduler found no valid assignment for {job_id}", job_id=job_id)
        return

    primary_id = schedule["primary"]
    shadow_ids = schedule["shadows"][: max(0, job.redundancy - 1)]
    vseed = int(np.random.randint(0, 2 ** 31))

    job.assigned_nodes  = [primary_id] + shadow_ids
    job.shadow_nodes    = shadow_ids
    job.status          = JobStatus.RUNNING
    job.verification_seed = vseed

    primary = nodes[primary_id]
    payload = {**job.dict(), "start_epoch": job.checkpoint_epoch, "verification_seed": vseed, "is_shadow": False}
    _log("schedule", f"Job {job_id} → {primary.name} (+{len(shadow_ids)} shadows)", job_id=job_id)

    try:
        requests.post(f"http://{primary.host}:{primary.port}/execute", json=payload, timeout=10)
    except Exception as exc:
        _log("error", f"Dispatch to {primary.name} failed: {exc}", job_id=job_id)
        job.status = JobStatus.FAILED

    # Dispatch shadow nodes
    for sid in shadow_ids:
        shadow = nodes[sid]
        shadow_payload = {**payload, "is_shadow": True}
        try:
            requests.post(f"http://{shadow.host}:{shadow.port}/execute", json=shadow_payload, timeout=10)
        except Exception as exc:
            _log("warn", f"Shadow {shadow.name} dispatch failed: {exc}", job_id=job_id)


# ---------------------------------------------------------------------------
# Routes

@app.get("/")
def root():
    return {
        "service": "NeuralMesh Coordinator",
        "nodes":   len(nodes),
        "jobs":    len(jobs),
    }


@app.post("/register")
def register_node(spec: NodeSpec):
    spec.last_heartbeat = datetime.utcnow().isoformat()
    nodes[spec.node_id] = spec
    mesh.add_node(spec.node_id, name=spec.name, ntype=spec.node_type)
    if not mesh.has_node("coordinator"):
        mesh.add_node("coordinator", name="Coordinator", ntype="coordinator")
    mesh.add_edge("coordinator", spec.node_id)
    ledger.create_account(spec.node_id, initial=100.0)
    _log("register", f"{spec.name} ({spec.node_type}) joined the mesh", node_id=spec.node_id)
    return {"node_id": spec.node_id, "status": "registered", "initial_credits": 100.0}


@app.post("/heartbeat")
def heartbeat(hb: HeartbeatRequest):
    node = nodes.get(hb.node_id)
    if not node:
        raise HTTPException(404, "Node not found — please re-register")

    now = datetime.utcnow().isoformat()
    node.last_heartbeat = now

    if node.status == NodeStatus.FAILED:
        node.status = NodeStatus.ACTIVE
        _log("recovery", f"{node.name} reconnected to mesh", node_id=hb.node_id)

    # Earn credits for idle compute contribution
    mult = ledger.get_multiplier(hb.node_id)
    earned = 0.0
    if hb.cpu_percent < 80:
        earned += 0.5 * mult
    if hb.gpu_percent is not None and hb.gpu_percent < 80:
        earned += 2.0 * mult
    if earned > 0:
        ledger.credit(hb.node_id, earned, f"Idle-time contribution (×{mult:.2f} streak multiplier)")

    return {
        "status":     "ok",
        "balance":    ledger.get_balance(hb.node_id),
        "streak":     ledger.get_streak(hb.node_id),
        "multiplier": ledger.get_multiplier(hb.node_id),
    }


@app.post("/submit")
def submit_job(job: TaskSpec, background_tasks: BackgroundTasks):
    cost = _estimate_cost(job)
    job.credit_cost = cost

    if not ledger.debit(job.submitter_id, cost, f"Job {job.job_id} ({job.task_type})"):
        balance = ledger.get_balance(job.submitter_id)
        raise HTTPException(402, f"Insufficient credits — need {cost}, have {balance:.2f}")

    jobs[job.job_id] = job
    _log("submit", f"Job {job.job_id} ({job.task_type}, {job.epochs} epochs) queued by {job.submitter_id}", job_id=job.job_id)
    background_tasks.add_task(_schedule_and_dispatch, job.job_id)
    return {"job_id": job.job_id, "status": "queued", "estimated_cost": cost}


@app.post("/checkpoint")
def receive_checkpoint(ckpt: CheckpointData):
    checkpoints[ckpt.job_id] = ckpt
    job = jobs.get(ckpt.job_id)
    if job:
        job.checkpoint_epoch = ckpt.epoch
    _log("checkpoint", f"Checkpoint saved for job {ckpt.job_id} at epoch {ckpt.epoch}", job_id=ckpt.job_id)
    return {"status": "saved", "epoch": ckpt.epoch}


@app.post("/result")
def receive_result(result: GradientResult):
    if result.job_id not in gradient_results:
        gradient_results[result.job_id] = []
    gradient_results[result.job_id].append(result)

    job = jobs.get(result.job_id)
    if not job:
        return {"status": "unknown_job"}

    all_results = gradient_results[result.job_id]
    needed = max(1, len(job.assigned_nodes))

    if len(all_results) >= needed or (len(all_results) >= 1 and not job.shadow_nodes):
        job.status = JobStatus.VERIFYING
        verified, report = gpow.verify(all_results, job.verification_seed or 0)

        if verified:
            job.status     = JobStatus.DONE
            job.result     = {"loss": round(result.loss, 6), "accuracy": round(result.accuracy, 4), "verified": True}
            job.completed_at = datetime.utcnow().isoformat()

            # Reward primary
            if job.assigned_nodes:
                pid = job.assigned_nodes[0]
                reward = job.credit_cost * 0.8
                ledger.credit(pid, reward, f"Job {job.job_id} completed")
                if pid in nodes:
                    nodes[pid].jobs_completed += 1
                    nodes[pid].trust_score = min(100.0, nodes[pid].trust_score + 0.5)
                    ledger._check_achievements(pid)

            # Reward shadows
            for sid in job.shadow_nodes:
                ledger.credit(sid, 5.0, f"Shadow GPoW for job {job.job_id}")
                if sid in nodes:
                    nodes[sid].trust_score = min(100.0, nodes[sid].trust_score + 0.2)

            _log("verify", f"✓ Job {job.job_id} verified — loss={result.loss:.4f} acc={result.accuracy:.2%}", job_id=job.job_id)

        else:
            _log("warn", f"✗ GPoW FAILED for job {job.job_id} — mismatched gradient hashes", job_id=job.job_id)
            for node_id in report.get("mismatched", []):
                ledger.slash(node_id, 30.0, f"GPoW mismatch on job {job.job_id}")
                if node_id in nodes:
                    nodes[node_id].trust_score = max(0.0, nodes[node_id].trust_score - 15.0)
                    _log("slash", f"Node {nodes[node_id].name} penalised — trust −15, credits −30", node_id=node_id)
            # Reschedule without guilty nodes
            job.status = JobStatus.QUEUED
            job.assigned_nodes = [n for n in job.assigned_nodes if n not in report.get("mismatched", [])]
            threading.Thread(target=_schedule_and_dispatch, args=(job.job_id,), daemon=True).start()

    return {"status": "received", "job_status": job.status}


# ---------------------------------------------------------------------------
# Query endpoints

@app.get("/nodes")
def list_nodes():
    return [
        {**n.dict(), "balance": ledger.get_balance(n.node_id), "streak": ledger.get_streak(n.node_id), "multiplier": ledger.get_multiplier(n.node_id)}
        for n in nodes.values()
    ]


@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    node = nodes.get(node_id)
    if not node:
        raise HTTPException(404)
    return {**node.dict(), "balance": ledger.get_balance(node_id), "streak": ledger.get_streak(node_id)}


@app.get("/jobs")
def list_jobs():
    return list(jobs.values())


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    return job


@app.get("/jobs/{job_id}/checkpoint")
def get_checkpoint(job_id: str):
    ckpt = checkpoints.get(job_id)
    if not ckpt:
        raise HTTPException(404, "No checkpoint saved yet")
    return ckpt


@app.get("/credits/{node_id}")
def get_credits(node_id: str):
    return {
        "node_id":      node_id,
        "balance":      ledger.get_balance(node_id),
        "streak":       ledger.get_streak(node_id),
        "multiplier":   ledger.get_multiplier(node_id),
        "achievements": ledger.get_achievements(node_id),
        "transactions": ledger.get_history(node_id, 30),
        "pledges":      ledger.get_pledges(node_id),
    }


@app.get("/leaderboard")
def leaderboard():
    rows = []
    for node_id, node in nodes.items():
        rows.append({
            "node_id":       node_id,
            "name":          node.name,
            "credits":       ledger.get_balance(node_id),
            "jobs_completed": node.jobs_completed,
            "trust_score":   node.trust_score,
            "streak_days":   ledger.get_streak(node_id),
            "multiplier":    ledger.get_multiplier(node_id),
            "achievements":  ledger.get_achievements(node_id),
        })
    return sorted(rows, key=lambda r: r["credits"], reverse=True)[:20]


@app.get("/activity")
def get_activity(limit: int = 50):
    return activity_log[:limit]


@app.get("/mesh")
def get_mesh():
    node_data = []
    for nid in mesh.nodes():
        nd = {"id": nid}
        if nid in nodes:
            nd["data"] = nodes[nid].dict()
        else:
            nd["data"] = mesh.nodes[nid]
        node_data.append(nd)
    return {
        "nodes": node_data,
        "edges": [{"source": u, "target": v} for u, v in mesh.edges()],
        "stats": {
            "node_count":  mesh.number_of_nodes(),
            "edge_count":  mesh.number_of_edges(),
            "density":     round(nx.density(mesh), 4) if mesh.number_of_nodes() > 1 else 0,
            "is_connected": nx.is_connected(mesh) if mesh.number_of_nodes() > 0 else False,
        },
    }


@app.get("/stats")
def get_stats():
    return {
        "active_nodes":                sum(1 for n in nodes.values() if n.status == NodeStatus.ACTIVE),
        "failed_nodes":                sum(1 for n in nodes.values() if n.status == NodeStatus.FAILED),
        "total_nodes":                 len(nodes),
        "running_jobs":                sum(1 for j in jobs.values() if j.status == JobStatus.RUNNING),
        "queued_jobs":                 sum(1 for j in jobs.values() if j.status == JobStatus.QUEUED),
        "completed_jobs":              sum(1 for j in jobs.values() if j.status == JobStatus.DONE),
        "failed_jobs":                 sum(1 for j in jobs.values() if j.status == JobStatus.FAILED),
        "total_checkpoints":           len(checkpoints),
        "total_credits_in_circulation": round(sum(ledger.get_all_balances().values()), 2),
    }


@app.get("/resource_history")
def get_resource_history():
    return resource_history[:60]
