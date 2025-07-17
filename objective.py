from __future__ import annotations
from typing import Tuple
import datetime
from tasks import Task
from providers import Provider
from utils import merge_intervals

DEFAULT_WEIGHTS: Tuple[float, float, float, float, float] = (
    1.0,    # a1 – total time
    1000.0, # a2 – budget penalty
    10000.0,# a3 – deadline penalty
    -0.1,   # b1 – provider profit (negative for minimisation)
    1.0     # b2 – provider idle penalty
)

def calc_objective(task: Task, scene_id: int, prov: Provider,
                   weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS) -> float:
    a1, a2, a3, b1, b2 = weights
    size = task.global_file_size + task.scene_size(scene_id)
    rate = min(task.bandwidth, prov.bandwidth)
    T_tx = size / rate
    T_cmp = task.scene_workload / prov.throughput
    T_tot = T_tx + T_cmp
    cost = T_tot * prov.price_per_gpu_hour
    profit_rate = prov.price_per_gpu_hour
    window_h = (task.deadline - task.start_time).total_seconds() / 3600.0
    idle = prov.idle_ratio()
    return (
        a1 * T_tot +
        a2 * max(0.0, cost - task.budget) +
        a3 * max(0.0, T_tot - window_h) +
        b1 * profit_rate +
        b2 * idle
    )