"""
NeuralMesh Task Executor
========================
Runs AI training jobs using pure NumPy.  No PyTorch or TensorFlow required.

Supported task types
--------------------
* train_ann  — 2-layer fully-connected network (ReLU + softmax)
* train_cnn  — simplified convolutional layer followed by FC (numpy conv2d)
* simulate   — Monte Carlo pi estimation (demonstrates non-ML workloads)
* preprocess — standardise + PCA on a random dataset

GPoW verification
-----------------
Every task receives a ``verification_seed`` from the coordinator.
Before training the executor draws a fixed verification mini-batch using
that seed and the INITIAL (untrained) model weights.  It then computes the
gradient and returns its SHA-256 hex digest.  Shadow nodes do the same.
Because numpy is deterministic, honest nodes produce identical hashes.
"""

import hashlib
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np


# ============================================================== ANN ============

class TwoLayerANN:
    """Minimal 2-layer ANN: input → ReLU → softmax."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, out_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    # ---- forward
    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X  = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(0.0, self._z1)              # ReLU
        z2       = self._a1 @ self.W2 + self.b2
        e        = np.exp(z2 - z2.max(axis=1, keepdims=True))
        self._p  = e / e.sum(axis=1, keepdims=True)       # softmax
        return self._p

    # ---- loss + accuracy
    def cross_entropy(self, y: np.ndarray) -> float:
        N = len(y)
        return -float(np.log(self._p[np.arange(N), y] + 1e-9).mean())

    def accuracy(self, y: np.ndarray) -> float:
        return float((self._p.argmax(1) == y).mean())

    # ---- backward (returns flattened gradient vector)
    def backward(self, y: np.ndarray, lr: float = 0.01) -> np.ndarray:
        N     = len(y)
        dp    = self._p.copy()
        dp[np.arange(N), y] -= 1.0
        dp   /= N
        dW2   = self._a1.T @ dp
        db2   = dp.sum(0)
        da1   = dp @ self.W2.T
        dz1   = da1 * (self._z1 > 0)
        dW1   = self._X.T @ dz1
        db1   = dz1.sum(0)
        if lr > 0:
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

    def weight_bytes(self) -> bytes:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2]).tobytes()


# ============================================================== CNN (simplified)

class SimpleCNN:
    """
    1 conv filter (3×3) → ReLU → global avg-pool → FC → softmax.
    Input: (N, H, W) grayscale images — H=W=8 for the demo.
    """

    def __init__(self, img_size: int, out_dim: int, seed: int = 0):
        rng     = np.random.RandomState(seed)
        self.F  = rng.randn(1, 3, 3).astype(np.float32) * 0.1   # 1 filter
        self.fc = TwoLayerANN(1, 16, out_dim, seed=seed + 1)
        self._img_size = img_size

    def forward(self, X: np.ndarray) -> np.ndarray:
        N, H, W = X.shape
        out_h, out_w = H - 2, W - 2
        conv = np.zeros((N, out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                conv[:, i, j] = (X[:, i:i+3, j:j+3] * self.F[0]).sum((1, 2))
        relu = np.maximum(0.0, conv)
        pooled = relu.mean(axis=(1, 2), keepdims=True).reshape(N, 1)
        return self.fc.forward(pooled)

    def cross_entropy(self, y): return self.fc.cross_entropy(y)
    def accuracy(self, y):      return self.fc.accuracy(y)
    def backward(self, y, lr=0.01): return self.fc.backward(y, lr)


# ============================================================== Executor

class TaskExecutor:

    def __init__(self, coordinator_url: str, node_state: dict):
        self._coord = coordinator_url
        self._state = node_state          # {"node_id": str | None}
        self.running_jobs: Dict[str, bool] = {}

    # ---------------------------------------------------------------- dispatch
    def run(self, req) -> Dict:
        self.running_jobs[req.job_id] = True
        try:
            if req.task_type == "simulate":
                return self._simulate(req)
            elif req.task_type == "preprocess":
                return self._preprocess(req)
            elif req.task_type == "train_cnn":
                return self._train_cnn(req)
            else:
                return self._train_ann(req)
        finally:
            self.running_jobs.pop(req.job_id, None)

    # ---------------------------------------------------------------- ANN
    def _train_ann(self, req) -> Dict:
        rng = np.random.RandomState(42)
        X = rng.randn(req.dataset_size, req.input_dim).astype(np.float32)
        y = rng.randint(0, req.output_dim, req.dataset_size)

        model = TwoLayerANN(req.input_dim, req.hidden_dim, req.output_dim, seed=0)

        # ---- GPoW verification gradient (BEFORE any training)
        vgrad_hash = self._compute_verification_gradient(model, X, y, req)

        lr = 0.01
        bs = req.batch_size
        final_loss, final_acc = 0.0, 0.0

        for epoch in range(req.start_epoch, req.start_epoch + req.epochs):
            idx = rng.permutation(req.dataset_size)
            ep_loss = 0.0
            for b in range(max(1, req.dataset_size // bs)):
                bi = idx[b * bs: (b + 1) * bs]
                model.forward(X[bi])
                model.backward(y[bi], lr=lr)
                ep_loss += model.cross_entropy(y[bi])
            ep_loss /= max(1, req.dataset_size // bs)
            model.forward(X)
            final_acc  = model.accuracy(y)
            final_loss = ep_loss

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                self._send_checkpoint(req.job_id, epoch + 1, model.weight_bytes())

            time.sleep(0.05)   # simulate real compute time

        return {"final_loss": float(final_loss), "final_accuracy": float(final_acc), "gradient_hash": vgrad_hash}

    # ---------------------------------------------------------------- CNN
    def _train_cnn(self, req) -> Dict:
        img_size = 8
        rng = np.random.RandomState(42)
        X = rng.randn(req.dataset_size, img_size, img_size).astype(np.float32)
        y = rng.randint(0, req.output_dim, req.dataset_size)

        model = SimpleCNN(img_size, req.output_dim, seed=0)

        # flatten for verification
        X_flat = X.reshape(req.dataset_size, img_size * img_size)
        fc_stub = TwoLayerANN(img_size * img_size, req.hidden_dim, req.output_dim, seed=0)
        vgrad_hash = self._compute_verification_gradient(fc_stub, X_flat, y, req)

        lr = 0.005
        final_loss, final_acc = 0.0, 0.0

        for epoch in range(req.start_epoch, req.start_epoch + req.epochs):
            bs = req.batch_size
            idx = rng.permutation(req.dataset_size)
            ep_loss = 0.0
            for b in range(max(1, req.dataset_size // bs)):
                bi = idx[b * bs: (b + 1) * bs]
                model.forward(X[bi])
                model.backward(y[bi], lr=lr)
                ep_loss += model.cross_entropy(y[bi])
            ep_loss /= max(1, req.dataset_size // bs)
            model.forward(X)
            final_acc  = model.accuracy(y)
            final_loss = ep_loss

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                self._send_checkpoint(req.job_id, epoch + 1, b"cnn_placeholder")

            time.sleep(0.05)

        return {"final_loss": float(final_loss), "final_accuracy": float(final_acc), "gradient_hash": vgrad_hash}

    # ---------------------------------------------------------------- Simulate
    def _simulate(self, req) -> Dict:
        """Monte Carlo π estimation — demonstrates non-ML workloads."""
        rng = np.random.RandomState(42)
        n   = req.dataset_size * req.epochs * 10
        pts = rng.rand(n, 2)
        pi_estimate = 4.0 * float((pts[:, 0]**2 + pts[:, 1]**2 < 1.0).mean())

        vseed_hash = hashlib.sha256(f"simulate:{req.verification_seed}:{pi_estimate:.6f}".encode()).hexdigest()
        return {"final_loss": abs(np.pi - pi_estimate), "final_accuracy": 1.0 - abs(np.pi - pi_estimate) / np.pi, "gradient_hash": vseed_hash}

    # ---------------------------------------------------------------- Preprocess
    def _preprocess(self, req) -> Dict:
        """Standardise + truncated SVD (pseudo-PCA) on a random dataset."""
        rng = np.random.RandomState(42)
        X   = rng.randn(req.dataset_size, req.input_dim).astype(np.float32)
        X   = (X - X.mean(0)) / (X.std(0) + 1e-8)

        # Truncated SVD via power iteration (no scipy)
        k   = min(10, req.input_dim, req.dataset_size)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        X_pca = U[:, :k] * S[:k]

        result_hash = hashlib.sha256(X_pca.astype(np.float32).tobytes()[:512]).hexdigest()
        return {"final_loss": 0.0, "final_accuracy": float(S[:k].sum() / (S.sum() + 1e-9)), "gradient_hash": result_hash}

    # ---------------------------------------------------------------- helpers
    def _compute_verification_gradient(
        self, model: TwoLayerANN, X: np.ndarray, y: np.ndarray, req
    ) -> str:
        """
        Compute gradient on the verification mini-batch using the
        INITIAL (untrained) model weights.  Returns SHA-256 hex.
        """
        seed = req.verification_seed or 0
        vrng = np.random.RandomState(seed)
        vidx = vrng.choice(len(X), min(req.batch_size, len(X)), replace=False)
        model.forward(X[vidx])
        vgrad = model.backward(y[vidx], lr=0.0)   # lr=0 → don't mutate weights
        return hashlib.sha256(vgrad.astype(np.float32).tobytes()).hexdigest()

    def _send_checkpoint(self, job_id: str, epoch: int, weight_bytes: bytes):
        try:
            import requests as req_lib
            weights_hash = hashlib.sha256(weight_bytes).hexdigest()
            req_lib.post(
                f"{self._coord}/checkpoint",
                json={"job_id": job_id, "node_id": self._state.get("node_id", "?"), "epoch": epoch, "weights_hash": weights_hash},
                timeout=5,
            )
        except Exception:
            pass   # checkpointing is best-effort


CHECKPOINT_INTERVAL = 2
