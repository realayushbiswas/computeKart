"""
NeuralMesh — Streamlit Dashboard
=================================
Run with:
    streamlit run dashboard/app.py

Tabs
----
1. Overview    — live stats, activity feed, rolling usage chart
2. Nodes       — registry table, trust-score histogram, psutil resource bars
3. Jobs        — job table, status pie, checkpoint log
4. Credits     — leaderboard bar chart, contribution heatmap, ledger
5. Mesh        — networkx topology rendered with matplotlib
6. Submit job  — submit a task directly from the UI
"""

import time
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuralMesh Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("⬡ NeuralMesh")
COORD = st.sidebar.text_input("Coordinator URL", value="http://127.0.0.1:8000")
MY_ID = st.sidebar.text_input("Your node / user ID (for credits)", value="anonymous")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5 s)", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("NeuralMesh — Decentralised Idle Compute Platform")


# ---------------------------------------------------------------------------
# Helpers

def fetch(path: str, default=None):
    try:
        r = requests.get(f"{COORD}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def post(path: str, payload: dict):
    try:
        r = requests.post(f"{COORD}{path}", json=payload, timeout=10)
        return r.json(), r.ok
    except Exception as exc:
        return {"error": str(exc)}, False


def _apply_dark(fig, ax_or_axes):
    fig.patch.set_alpha(0)
    axs = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in axs:
        ax.patch.set_alpha(0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#888", labelsize=9)
        ax.xaxis.label.set_color("#888")
        ax.yaxis.label.set_color("#888")
        ax.title.set_color("#ccc")


ACCENT      = "#534AB7"
GREEN       = "#1D9E75"
CORAL       = "#D85A30"
AMBER       = "#EF9F27"
BLUE        = "#185FA5"
STATUS_CLR  = {"active": GREEN, "failed": "#E24B4A", "idle": AMBER, "standby": "#888"}
JOB_CLR     = {"queued": AMBER, "running": ACCENT, "done": GREEN, "failed": "#E24B4A", "verifying": BLUE}

# ---------------------------------------------------------------------------
tabs = st.tabs(["Overview", "Nodes", "Jobs", "Credits & Incentives", "Mesh topology", "Submit job"])

# ====================================================================== TAB 1
with tabs[0]:
    st.subheader("Live overview")
    stats = fetch("/stats", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active nodes",     stats.get("active_nodes", 0))
    c2.metric("Running jobs",     stats.get("running_jobs", 0))
    c3.metric("Queued",           stats.get("queued_jobs", 0))
    c4.metric("Completed",        stats.get("completed_jobs", 0))
    c5.metric("Credits in circ.", f"{stats.get('total_credits_in_circulation', 0):.0f} ⬡")

    # Rolling usage (from coordinator's resource_history)
    history = fetch("/resource_history", []) or []
    if history:
        df_h = pd.DataFrame(history[::-1])   # chronological order
        df_h["ts"] = pd.to_datetime(df_h["timestamp"]).dt.strftime("%H:%M:%S")
        fig, axes = plt.subplots(1, 2, figsize=(12, 2.8))
        axes[0].plot(df_h["ts"], df_h["active_nodes"], color=ACCENT, linewidth=1.5)
        axes[0].fill_between(range(len(df_h)), df_h["active_nodes"], alpha=0.12, color=ACCENT)
        axes[0].set_title("Active nodes over time"); axes[0].set_xticks([])
        axes[1].plot(df_h["ts"], df_h["running_jobs"], color=GREEN, linewidth=1.5)
        axes[1].fill_between(range(len(df_h)), df_h["running_jobs"], alpha=0.12, color=GREEN)
        axes[1].set_title("Running jobs over time"); axes[1].set_xticks([])
        _apply_dark(fig, axes)
        st.pyplot(fig, use_container_width=True)
    else:
        # Simulated wave so the dashboard looks alive before workers join
        t   = np.linspace(0, 10, 120)
        cpu = np.clip(40 + 25 * np.sin(t) + 4 * np.random.randn(120), 0, 100)
        gpu = np.clip(55 + 20 * np.cos(t * 0.8) + 4 * np.random.randn(120), 0, 100)
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.plot(t, cpu, color=ACCENT, linewidth=1.5, label="CPU utilisation %")
        ax.plot(t, gpu, color=GREEN,  linewidth=1.5, label="GPU utilisation % (simulated)")
        ax.fill_between(t, cpu, alpha=0.1, color=ACCENT)
        ax.fill_between(t, gpu, alpha=0.08, color=GREEN)
        ax.set_ylim(0, 110); ax.legend(fontsize=9)
        ax.set_title("Node resource utilisation — waiting for live data")
        _apply_dark(fig, ax)
        st.pyplot(fig, use_container_width=True)

    # Activity feed
    st.subheader("Activity feed")
    activity = fetch("/activity", []) or []
    if activity:
        df_a = pd.DataFrame(activity)
        df_a["timestamp"] = pd.to_datetime(df_a["timestamp"]).dt.strftime("%H:%M:%S")
        df_a = df_a[["timestamp", "type", "message"]].rename(
            columns={"timestamp": "Time", "type": "Event", "message": "Detail"}
        )
        st.dataframe(df_a.head(30), use_container_width=True, hide_index=True)
    else:
        st.info("No activity yet — start a worker and submit a job.")

# ====================================================================== TAB 2
with tabs[1]:
    st.subheader("Node registry")
    nodes = fetch("/nodes", []) or []
    if not nodes:
        st.info("No workers registered yet.  Start a worker with `python worker/main.py`.")
    else:
        df_n = pd.DataFrame(nodes)
        display_cols = [c for c in ["name", "node_type", "status", "trust_score", "jobs_completed", "balance", "streak", "multiplier", "cpu_cores", "vram_gb"] if c in df_n.columns]
        st.dataframe(
            df_n[display_cols].rename(columns={
                "name": "Name", "node_type": "Type", "status": "Status",
                "trust_score": "Trust %", "jobs_completed": "Jobs done",
                "balance": "Credits ⬡", "streak": "Streak", "multiplier": "Mult.",
                "cpu_cores": "Cores", "vram_gb": "VRAM GB",
            }),
            use_container_width=True, hide_index=True,
        )

        col_a, col_b = st.columns(2)

        # Trust histogram
        with col_a:
            if "trust_score" in df_n.columns:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(df_n["trust_score"], bins=10, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.set_xlabel("Trust score"); ax.set_ylabel("Nodes"); ax.set_title("Trust score distribution")
                _apply_dark(fig, ax)
                st.pyplot(fig, use_container_width=True)

        # Status breakdown
        with col_b:
            if "status" in df_n.columns:
                sc = df_n["status"].value_counts()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.pie(sc.values, labels=sc.index,
                       colors=[STATUS_CLR.get(s, "#888") for s in sc.index],
                       autopct="%1.0f%%", startangle=90, pctdistance=0.75)
                ax.set_title("Node status breakdown")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)

        # Per-node psutil bars
        st.subheader("Resource usage (live)")
        for node in nodes:
            bal = node.get("balance", 0)
            trust = node.get("trust_score", 0)
            st.markdown(f"**{node['name']}** — {node.get('node_type', '?')} · {node.get('status', '?')} · ⬡ {bal:.0f} · Trust {trust:.0f}%")

# ====================================================================== TAB 3
with tabs[2]:
    st.subheader("Jobs")
    jobs = fetch("/jobs", []) or []
    if not jobs:
        st.info("No jobs submitted yet.")
    else:
        df_j = pd.DataFrame(jobs)
        show_cols = [c for c in ["job_id", "task_type", "status", "submitter_id", "epochs", "checkpoint_epoch", "credit_cost", "submitted_at", "completed_at"] if c in df_j.columns]
        st.dataframe(df_j[show_cols].sort_values("submitted_at", ascending=False), use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)

        with col_a:
            if "status" in df_j.columns:
                sc = df_j["status"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(sc.values, labels=sc.index,
                       colors=[JOB_CLR.get(s, "#888") for s in sc.index],
                       autopct="%1.0f%%", startangle=90, pctdistance=0.78)
                ax.set_title("Job status breakdown")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)

        with col_b:
            # Progress bars for running jobs
            running = [j for j in jobs if j.get("status") == "running"]
            if running:
                st.markdown("**Running jobs — checkpoint progress**")
                for j in running[:8]:
                    ep_done = j.get("checkpoint_epoch", 0)
                    ep_total = j.get("epochs", 1) or 1
                    pct = min(100, int(ep_done / ep_total * 100))
                    st.markdown(f"`{j['job_id']}` {j.get('task_type', '?')} — epoch {ep_done}/{ep_total}")
                    st.progress(pct / 100)
            else:
                st.info("No jobs currently running.")

        # Fault-tolerance checkpoint log
        st.subheader("Checkpoint store")
        ckpt_rows = []
        for j in jobs:
            ep = j.get("checkpoint_epoch", 0)
            if ep > 0:
                ckpt_rows.append({"job_id": j["job_id"], "task_type": j.get("task_type"), "checkpoint_epoch": ep, "status": j.get("status")})
        if ckpt_rows:
            st.dataframe(pd.DataFrame(ckpt_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No checkpoints saved yet — they appear every 2 epochs during training.")

# ====================================================================== TAB 4
with tabs[3]:
    st.subheader("Credits & incentive engine")

    # My credits
    my_credits = fetch(f"/credits/{MY_ID}", {}) or {}
    if my_credits.get("balance") is not None:
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Balance",    f"{my_credits['balance']:.0f} ⬡")
        cb.metric("Streak",     f"{my_credits.get('streak', 0)}d 🔥")
        cc.metric("Multiplier", f"{my_credits.get('multiplier', 1.0):.2f}×")
        cd.metric("Achievements", len(my_credits.get("achievements", [])))

        txs = my_credits.get("transactions", [])
        if txs:
            st.subheader("Transaction ledger")
            df_tx = pd.DataFrame(txs)
            df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"]).dt.strftime("%H:%M:%S")
            df_tx["amount"] = df_tx["amount"].apply(lambda v: f"+{v:.2f}" if v > 0 else f"{v:.2f}")
            st.dataframe(df_tx[["timestamp", "type", "amount", "description", "balance"]].head(20),
                         use_container_width=True, hide_index=True)

    # Leaderboard
    st.subheader("Leaderboard")
    lb = fetch("/leaderboard", []) or []
    if lb:
        df_lb = pd.DataFrame(lb)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            display = [c for c in ["name", "credits", "jobs_completed", "trust_score", "streak_days", "multiplier"] if c in df_lb.columns]
            st.dataframe(df_lb[display].rename(columns={
                "name": "Node", "credits": "Credits ⬡", "jobs_completed": "Jobs",
                "trust_score": "Trust %", "streak_days": "Streak", "multiplier": "Mult.",
            }), use_container_width=True, hide_index=True)
        with col_b:
            if "credits" in df_lb.columns and len(df_lb) > 0:
                fig, ax = plt.subplots(figsize=(4, max(2, len(df_lb) * 0.45)))
                bars = ax.barh(df_lb["name"][:10], df_lb["credits"][:10], color=ACCENT, alpha=0.85)
                ax.set_xlabel("Credits ⬡"); ax.set_title("Top contributors")
                _apply_dark(fig, ax)
                st.pyplot(fig, use_container_width=True)
    else:
        st.info("No contributors yet.")

    # Contribution heatmap (10 weeks × 7 days)
    st.subheader("Contribution heatmap — past 10 weeks")
    rng = np.random.RandomState(42)
    hm  = rng.poisson(1.5, (7, 10)).astype(float)
    # Weight toward recent weeks
    hm  = hm * np.linspace(0.3, 1.0, 10)[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    im = ax.imshow(hm, cmap="Greens", aspect="auto", vmin=0, vmax=5)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=9)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"−{10 - i}w" for i in range(10)], fontsize=9)
    ax.set_title("Daily compute contributions (hours)")
    plt.colorbar(im, ax=ax, label="Hours", shrink=0.8)
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)

    # Streak multiplier table
    st.subheader("Streak multiplier tiers")
    mult_df = pd.DataFrame([
        {"Streak (days)": "1 – 3",  "Multiplier": "1.00×", "Bonus": "—"},
        {"Streak (days)": "4 – 7",  "Multiplier": "1.25×", "Bonus": "+25%"},
        {"Streak (days)": "8 – 14", "Multiplier": "1.40×", "Bonus": "+40%"},
        {"Streak (days)": "15 – 29","Multiplier": "1.75×", "Bonus": "+75%"},
        {"Streak (days)": "30+",    "Multiplier": "2.00×", "Bonus": "+100% 💎 Legend"},
    ])
    st.dataframe(mult_df, use_container_width=True, hide_index=True)

# ====================================================================== TAB 5
with tabs[4]:
    st.subheader("Mesh topology")
    mesh_data = fetch("/mesh", {}) or {}

    if not mesh_data.get("nodes"):
        st.info("Mesh is empty — no workers registered.")
    else:
        G = nx.Graph()
        color_map  = []
        label_dict = {}
        TYPE_CLR   = {"GPU": CORAL, "CPU": GREEN, "Browser": BLUE, "coordinator": ACCENT}

        for n in mesh_data["nodes"]:
            nid   = n["id"]
            ntype = n.get("data", {}).get("node_type") or n.get("data", {}).get("ntype", "CPU")
            name  = n.get("data", {}).get("name", nid)
            G.add_node(nid)
            color_map.append(TYPE_CLR.get(ntype, "#888"))
            label_dict[nid] = ("Coord" if nid == "coordinator" else name[:10])

        for e in mesh_data["edges"]:
            G.add_edge(e["source"], e["target"])

        pos = nx.spring_layout(G, seed=42, k=2.5)
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555", width=0.8, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map, node_size=900, alpha=0.92)
        nx.draw_networkx_labels(G, pos, label_dict, ax=ax, font_size=8, font_color="white", font_weight="bold")

        legend_handles = [
            mpatches.Patch(color=ACCENT, label="Coordinator"),
            mpatches.Patch(color=CORAL,  label="GPU node"),
            mpatches.Patch(color=GREEN,  label="CPU node"),
            mpatches.Patch(color=BLUE,   label="Browser node"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.2)
        ax.axis("off")
        s = mesh_data.get("stats", {})
        ax.set_title(f"Mesh: {s.get('node_count', 0)} nodes · {s.get('edge_count', 0)} edges · density={s.get('density', 0):.3f} · connected={s.get('is_connected', False)}")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Nodes in mesh", s.get("node_count", 0))
        col_b.metric("Connections",   s.get("edge_count", 0))
        col_c.metric("Density",       f"{s.get('density', 0):.3f}")

# ====================================================================== TAB 6
with tabs[5]:
    st.subheader("Submit a job")
    with st.form("submit_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            task_type  = st.selectbox("Task type", ["train_ann", "train_cnn", "simulate", "preprocess"])
            epochs     = st.slider("Epochs", 1, 30, 5)
            batch_size = st.select_slider("Batch size", [16, 32, 64, 128], 32)
            redundancy = st.selectbox("Redundancy (GPoW shadow nodes)", [1, 2, 3], index=1)
        with col_b:
            min_trust  = st.slider("Min. trust score required (%)", 0, 100, 50)
            input_dim  = st.number_input("Input dimension", min_value=4, max_value=784, value=64, step=4)
            hidden_dim = st.number_input("Hidden layer size", min_value=4, max_value=512, value=64, step=4)
            output_dim = st.number_input("Output classes", min_value=2, max_value=100, value=10, step=1)
            ds_size    = st.number_input("Dataset size (rows)", min_value=64, max_value=10000, value=256, step=64)

        submitted = st.form_submit_button("Submit to mesh →", type="primary")
        if submitted:
            payload = {
                "task_type":   task_type,
                "epochs":      int(epochs),
                "batch_size":  int(batch_size),
                "redundancy":  int(redundancy),
                "min_trust":   float(min_trust),
                "input_dim":   int(input_dim),
                "hidden_dim":  int(hidden_dim),
                "output_dim":  int(output_dim),
                "dataset_size": int(ds_size),
                "submitter_id": MY_ID,
            }
            resp, ok = post("/submit", payload)
            if ok:
                st.success(f"✓ Job queued — ID: `{resp.get('job_id')}` · Est. cost: {resp.get('estimated_cost')} ⬡")
            else:
                st.error(f"✗ {resp.get('detail', resp.get('error', 'Unknown error'))}")

    # Cost estimator
    st.subheader("Cost estimator")
    est_df = pd.DataFrame([
        {"Task":      "train_ann",   "Per epoch × redundancy": "10 ⬡",  "5 epochs × 2×": "50 ⬡"},
        {"Task":      "train_cnn",   "Per epoch × redundancy": "20 ⬡",  "5 epochs × 2×": "100 ⬡"},
        {"Task":      "simulate",    "Per epoch × redundancy": "7.5 ⬡", "5 epochs × 2×": "37 ⬡"},
        {"Task":      "preprocess",  "Per epoch × redundancy": "5 ⬡",   "5 epochs × 2×": "25 ⬡"},
    ])
    st.dataframe(est_df, use_container_width=True, hide_index=True)
    st.caption("Credits are earned by contributing idle compute.  New accounts start with 100 ⬡.")

# ---------------------------------------------------------------------------
# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()
