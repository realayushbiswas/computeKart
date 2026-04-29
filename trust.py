import os
from typing import Dict, List, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.hmac import HMAC


class GPoWEngine:
    """
    Gradient Proof-of-Work verification engine.

    The coordinator injects the same (seed, mini-batch) into every worker
    (primary + shadows). After training, each worker computes the gradient
    of the INITIAL model on that fixed verification batch and returns its
    SHA-256 hex digest. This engine compares the digests.

    Because numpy training is deterministic given the same RNG seed, the
    verification gradient at any fixed model state is identical across all
    honest nodes.  A mismatching hash therefore indicates either a tampered
    result or a faulty implementation — both warrant penalisation.
    """

    TOLERANCE_LOSS = 0.05   # allowed loss divergence before flagging

    def __init__(self):
        self._secret: bytes = os.urandom(32)

    # ------------------------------------------------------------------ public
    def sign_task(self, payload: str) -> str:
        """HMAC-SHA256 of a task payload — prevents task tampering in transit."""
        h = HMAC(self._secret, hashes.SHA256(), backend=default_backend())
        h.update(payload.encode())
        return h.finalize().hex()

    def verify_signature(self, payload: str, signature: str) -> bool:
        """Verify a previously-signed task payload."""
        expected = self.sign_task(payload)
        return expected == signature

    def compute_challenge_hash(self, seed: int, node_id: str) -> str:
        """
        Derive a per-node challenge nonce from the verification seed.
        Workers receive this nonce along with the seed to prevent pre-computation.
        """
        h = HMAC(self._secret, hashes.SHA256(), backend=default_backend())
        h.update(seed.to_bytes(8, "big") + node_id.encode())
        return h.finalize().hex()[:16]

    def verify(self, results: List, verification_seed: int) -> Tuple[bool, Dict]:
        """
        Compare gradient hashes across primary and shadow nodes.

        Args:
            results: list of GradientResult objects
            verification_seed: the seed used to generate the verification batch

        Returns:
            (verified: bool, report: dict)
        """
        if not results:
            return False, {"reason": "no_results"}

        primary_results = [r for r in results if not r.is_shadow]
        shadow_results  = [r for r in results if r.is_shadow]

        if not primary_results:
            return False, {"reason": "no_primary_result"}

        primary = primary_results[0]

        # No shadow nodes submitted — accept in low-security mode
        if not shadow_results:
            return True, {
                "mode": "no_shadow",
                "primary": primary.node_id,
                "primary_hash": primary.gradient_hash[:16] + "...",
                "note": "Single-node job; GPoW skipped.",
            }

        mismatched_nodes: List[str] = []
        details: List[Dict] = []

        for shadow in shadow_results:
            hash_match = shadow.gradient_hash == primary.gradient_hash
            loss_diff  = abs(shadow.loss - primary.loss)
            loss_ok    = loss_diff <= self.TOLERANCE_LOSS

            verdict = "ok" if (hash_match and loss_ok) else "MISMATCH"
            if verdict != "ok":
                mismatched_nodes.append(shadow.node_id)

            details.append({
                "shadow_node":  shadow.node_id,
                "hash_match":   hash_match,
                "loss_diff":    round(loss_diff, 6),
                "loss_ok":      loss_ok,
                "verdict":      verdict,
            })

        verified = len(mismatched_nodes) == 0
        return verified, {
            "verified":       verified,
            "primary":        primary.node_id,
            "primary_hash":   primary.gradient_hash[:24] + "...",
            "shadows":        [s.node_id for s in shadow_results],
            "mismatched":     mismatched_nodes,
            "details":        details,
            "seed_used":      verification_seed,
        }
