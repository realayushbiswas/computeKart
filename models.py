import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    GPU = "GPU"
    CPU = "CPU"
    BROWSER = "Browser"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    FAILED = "failed"
    STANDBY = "standby"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"


class NodeSpec(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    host: str = "localhost"
    port: int = 8001
    node_type: NodeType = NodeType.CPU
    vram_gb: float = 0.0
    cpu_cores: int = 4
    trust_score: float = 50.0
    status: NodeStatus = NodeStatus.ACTIVE
    registered_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: Optional[str] = None
    jobs_completed: int = 0
    streak_days: int = 0
    multiplier: float = 1.0


class TaskSpec(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = "train_ann"
    epochs: int = 5
    batch_size: int = 32
    min_trust: float = 40.0
    redundancy: int = 2
    dataset_size: int = 512
    input_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 10
    submitter_id: str = "anonymous"
    status: JobStatus = JobStatus.QUEUED
    assigned_nodes: List[str] = []
    shadow_nodes: List[str] = []
    checkpoint_epoch: int = 0
    result: Optional[Dict] = None
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    credit_cost: float = 0.0
    verification_seed: Optional[int] = None


class HeartbeatRequest(BaseModel):
    node_id: str
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float] = None
    active_jobs: int = 0


class CheckpointData(BaseModel):
    job_id: str
    node_id: str
    epoch: int
    weights_hash: str


class GradientResult(BaseModel):
    job_id: str
    node_id: str
    epoch: int
    gradient_hash: str
    loss: float
    accuracy: float
    is_shadow: bool = False


class ScheduleResult(BaseModel):
    primary: str
    shadows: List[str] = []
