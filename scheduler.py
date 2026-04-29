"""
Fair-share scheduler for NeuralMesh using integer linear programming (PuLP).

Objective: maximise the trust-weighted coverage of selected nodes while
enforcing Dominant Resource Fairness (DRF) — nodes that have already earned
many credits are penalised slightly so that credit-poor contributors get work.

Constraints
-----------
* Exactly 1 primary node per job.
* Up to (redundancy − 1) shadow nodes per job.
* A node cannot be both primary and shadow.
* Nodes below min_trust are ineligible.
* Nodes with status != "active" are ineligible.

Fallback
--------
If PuLP fails for any reason (solver unavailable, infeasible) the scheduler
falls back to a simple greedy selection by trust score.
"""

from typing import Dict, List, Optional

import pulp


def fair_schedule(job, available_nodes: List, balances: Dict[str, float]) -> Optional[Dict]:
    """
    Parameters
    ----------
    job            : TaskSpec — the job to schedule
    available_nodes: list of NodeSpec objects that are currently active
    balances       : dict[node_id -> current_credits]

    Returns
    -------
    {"primary": node_id, "shadows": [node_id, ...]}  or  None if infeasible
    """
    eligible = [
        n for n in available_nodes
        if n.trust_score >= job.min_trust and n.status == "active"
    ]
    if not eligible:
        return None

    n_nodes  = len(eligible)
    node_ids = [n.node_id for n in eligible]

    # Weights for the objective
    max_balance = max(balances.values(), default=1.0) or 1.0
    trust_w = [n.trust_score / 100.0 for n in eligible]
    # DRF fairness: prefer nodes with *lower* accumulated balance
    fair_w  = [
        1.0 - (balances.get(nid, 0.0) / max_balance) * 0.30
        for nid in node_ids
    ]

    n_shadows = max(0, job.redundancy - 1)

    try:
        prob = pulp.LpProblem("NeuralMesh_Schedule", pulp.LpMaximize)

        x = [pulp.LpVariable(f"primary_{i}", cat="Binary") for i in range(n_nodes)]
        s = [pulp.LpVariable(f"shadow_{i}",  cat="Binary") for i in range(n_nodes)]

        # Objective: maximise combined trust × fairness weight
        prob += pulp.lpSum(
            (trust_w[i] * fair_w[i]) * x[i] + 0.5 * (trust_w[i] * fair_w[i]) * s[i]
            for i in range(n_nodes)
        )

        # Exactly one primary
        prob += pulp.lpSum(x) == 1

        # Shadow count: at most n_shadows, at least min(1, n_shadows) if possible
        prob += pulp.lpSum(s) <= n_shadows
        if n_shadows > 0 and n_nodes > 1:
            prob += pulp.lpSum(s) >= min(1, n_shadows)

        for i in range(n_nodes):
            # A node cannot serve as both primary and shadow simultaneously
            prob += x[i] + s[i] <= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        primary_id  = None
        shadow_ids  = []

        if pulp.LpStatus[prob.status] in ("Optimal", "Feasible"):
            for i in range(n_nodes):
                if pulp.value(x[i]) and pulp.value(x[i]) > 0.5:
                    primary_id = node_ids[i]
                if pulp.value(s[i]) and pulp.value(s[i]) > 0.5:
                    shadow_ids.append(node_ids[i])

        if primary_id:
            return {"primary": primary_id, "shadows": shadow_ids}

    except Exception:
        pass   # fall through to greedy

    # --- greedy fallback ---
    ranked = sorted(
        eligible,
        key=lambda n: n.trust_score - 0.1 * (balances.get(n.node_id, 0) / max_balance),
        reverse=True,
    )
    primary_id = ranked[0].node_id
    shadow_ids = [r.node_id for r in ranked[1: 1 + n_shadows]]
    return {"primary": primary_id, "shadows": shadow_ids}
