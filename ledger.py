from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional


STREAK_MULTIPLIERS = [
    (30, 2.0),
    (15, 1.75),
    (8,  1.40),
    (4,  1.25),
    (1,  1.00),
]

ACHIEVEMENT_DEFS = {
    "first_job":     {"icon": "⚡", "name": "First job",     "desc": "Submit first task"},
    "week_warrior":  {"icon": "🔥", "name": "Week warrior",  "desc": "7-day streak"},
    "gpu_guru":      {"icon": "🧠", "name": "GPU guru",      "desc": "Share 100 GPU-hours"},
    "top_100":       {"icon": "🏆", "name": "Top 100",       "desc": "Reach rank #100"},
    "trusted":       {"icon": "🔒", "name": "Trusted",       "desc": "Trust score ≥ 95%"},
    "night_owl":     {"icon": "🌙", "name": "Night owl",     "desc": "Contribute 2–6 AM"},
    "researcher":    {"icon": "⚗️",  "name": "Researcher",   "desc": "Run 20 jobs"},
    "shardmaster":   {"icon": "🚀", "name": "Shardmaster",   "desc": "Job spans 5+ nodes"},
    "legend":        {"icon": "💎", "name": "Legend",        "desc": "30-day streak"},
    "federated":     {"icon": "🤝", "name": "Federated",    "desc": "Run federated job"},
}


def _multiplier_for_streak(streak_days: int) -> float:
    for min_days, mult in STREAK_MULTIPLIERS:
        if streak_days >= min_days:
            return mult
    return 1.0


class CreditLedger:
    def __init__(self):
        self._balances: Dict[str, float] = {}
        self._history: Dict[str, List[Dict]] = {}
        self._streaks: Dict[str, int] = {}
        self._last_active: Dict[str, str] = {}
        self._achievements: Dict[str, List[str]] = {}
        self._compute_pledges: Dict[str, List[Dict]] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------ account
    def create_account(self, node_id: str, initial: float = 100.0):
        with self._lock:
            if node_id not in self._balances:
                self._balances[node_id] = initial
                self._history[node_id] = []
                self._streaks[node_id] = 0
                self._achievements[node_id] = []
                self._compute_pledges[node_id] = []
                if initial > 0:
                    self._history[node_id].append(self._tx("credit", initial, "Welcome bonus", initial))

    # ------------------------------------------------------------------ credits
    def credit(self, node_id: str, amount: float, description: str = ""):
        with self._lock:
            self._ensure(node_id)
            self._balances[node_id] += amount
            bal = self._balances[node_id]
            self._history[node_id].insert(0, self._tx("credit", round(amount, 2), description, round(bal, 2)))
            self._update_streak(node_id)
            self._check_achievements(node_id)

    def debit(self, node_id: str, amount: float, description: str = "") -> bool:
        with self._lock:
            self._ensure(node_id)
            if self._balances.get(node_id, 0) < amount:
                return False
            self._balances[node_id] -= amount
            bal = self._balances[node_id]
            self._history[node_id].insert(0, self._tx("debit", -round(amount, 2), description, round(bal, 2)))
            return True

    def slash(self, node_id: str, amount: float, reason: str = "GPoW violation"):
        """Penalize a node — resets streak."""
        with self._lock:
            self._ensure(node_id)
            actual = min(amount, self._balances.get(node_id, 0))
            self._balances[node_id] -= actual
            self._streaks[node_id] = 0
            bal = self._balances[node_id]
            self._history[node_id].insert(0, self._tx("slash", -round(actual, 2), reason, round(bal, 2)))

    # ------------------------------------------------------------------ pledges
    def add_pledge(self, node_id: str, window_start: str, window_end: str, resource: str):
        with self._lock:
            self._ensure(node_id)
            self._compute_pledges[node_id].append({
                "start": window_start,
                "end": window_end,
                "resource": resource,
                "status": "pending",
                "bonus_pct": 20,
            })

    def fulfil_pledge(self, node_id: str, pledge_idx: int, base_earned: float):
        with self._lock:
            pledges = self._compute_pledges.get(node_id, [])
            if pledge_idx < len(pledges):
                pledges[pledge_idx]["status"] = "fulfilled"
                bonus = base_earned * 0.20
                self._balances[node_id] = self._balances.get(node_id, 0) + bonus
                bal = self._balances[node_id]
                self._history[node_id].insert(0, self._tx("credit", round(bonus, 2), "Pledge bonus +20%", round(bal, 2)))

    def get_pledges(self, node_id: str) -> List[Dict]:
        return list(self._compute_pledges.get(node_id, []))

    # ------------------------------------------------------------------ queries
    def get_balance(self, node_id: str) -> float:
        return round(self._balances.get(node_id, 0), 2)

    def get_all_balances(self) -> Dict[str, float]:
        return dict(self._balances)

    def get_history(self, node_id: str, limit: int = 30) -> List[Dict]:
        return list(self._history.get(node_id, []))[:limit]

    def get_streak(self, node_id: str) -> int:
        return self._streaks.get(node_id, 0)

    def get_multiplier(self, node_id: str) -> float:
        return _multiplier_for_streak(self._streaks.get(node_id, 0))

    def get_achievements(self, node_id: str) -> List[str]:
        return list(self._achievements.get(node_id, []))

    def get_all_achievements_meta(self) -> Dict:
        return ACHIEVEMENT_DEFS

    # ------------------------------------------------------------------ helpers
    def _ensure(self, node_id: str):
        if node_id not in self._balances:
            self._balances[node_id] = 0.0
            self._history[node_id] = []
            self._streaks[node_id] = 0
            self._achievements[node_id] = []
            self._compute_pledges[node_id] = []

    @staticmethod
    def _tx(kind: str, amount: float, desc: str, balance: float) -> Dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "type": kind,
            "amount": amount,
            "description": desc,
            "balance": balance,
        }

    def _update_streak(self, node_id: str):
        now = datetime.utcnow().isoformat()
        last = self._last_active.get(node_id)
        if last:
            diff = (datetime.utcnow() - datetime.fromisoformat(last)).total_seconds()
            if diff < 90000:   # within 25 hours (grace period)
                self._streaks[node_id] = self._streaks.get(node_id, 0) + 1
            elif diff > 172800:  # more than 2 days — reset
                self._streaks[node_id] = 1
        else:
            self._streaks[node_id] = max(1, self._streaks.get(node_id, 0))
        self._last_active[node_id] = now

    def _check_achievements(self, node_id: str):
        unlocked = set(self._achievements.get(node_id, []))
        streak = self._streaks.get(node_id, 0)
        if streak >= 7 and "week_warrior" not in unlocked:
            unlocked.add("week_warrior")
        if streak >= 30 and "legend" not in unlocked:
            unlocked.add("legend")
        bal = self._balances.get(node_id, 0)
        if bal >= 500 and "gpu_guru" not in unlocked:
            unlocked.add("gpu_guru")
        self._achievements[node_id] = list(unlocked)
