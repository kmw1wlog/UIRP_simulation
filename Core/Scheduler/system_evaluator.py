from __future__ import annotations
import datetime as dt
from typing import Dict, Any

from Model.tasks import Tasks
from Model.providers import Providers


def evaluate(tasks: Tasks, providers: Providers) -> Dict[str, Any]:
    """Aggregate metrics for completed schedule.

    Parameters
    ----------
    tasks: Tasks
        Collection of tasks after scheduling.
    providers: Providers
        Providers whose ``schedule`` fields have been populated.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing per‑task and system‑level metrics.
    """
    task_stats: Dict[str, Dict[str, Any]] = {
        t.id: {"cost": 0.0, "start": None, "finish": None} for t in tasks
    }

    # ---- per task aggregation ----
    for p_idx, prov in enumerate(providers):
        for t_id, _scene, st, ft in getattr(prov, "schedule", []):
            dur_h = (ft - st).total_seconds() / 3600.0
            cost = dur_h * prov.price_per_gpu_hour
            rec = task_stats[t_id]
            rec["cost"] += cost
            if rec["start"] is None or st < rec["start"]:
                rec["start"] = st
            if rec["finish"] is None or ft > rec["finish"]:
                rec["finish"] = ft

    deadline_hits = 0
    deadline_misses = 0
    lateness_vals = []

    starts = []
    finishes = []
    for t in tasks:
        rec = task_stats[t.id]
        start = rec["start"]
        finish = rec["finish"]
        cost = rec["cost"]

        budget_over = max(0.0, cost - t.budget)
        rec["budget_overrun"] = budget_over
        if start is not None:
            starts.append(start)
        if finish is not None:
            finishes.append(finish)

        if finish is not None:
            lateness = max(0.0, (finish - t.deadline).total_seconds() / 3600.0)
            rec["lateness_hours"] = lateness
            hit = lateness == 0.0
            rec["deadline_hit"] = hit
            if hit:
                deadline_hits += 1
            else:
                deadline_misses += 1
            lateness_vals.append(lateness)
        else:
            rec["lateness_hours"] = None
            rec["deadline_hit"] = False
            deadline_misses += 1

    if starts and finishes:
        makespan_h = (max(finishes) - min(starts)).total_seconds() / 3600.0
        throughput = len(finishes) / makespan_h if makespan_h > 0 else 0.0
    else:
        makespan_h = 0.0
        throughput = 0.0

    avg_lateness = sum(lateness_vals) / len(lateness_vals) if lateness_vals else 0.0

    prov_util = {
        idx: 1.0 - prov.idle_ratio() for idx, prov in enumerate(providers)
    }

    return {
        "tasks": task_stats,
        "makespan_hours": makespan_h,
        "throughput_tasks_per_hour": throughput,
        "deadline_hits": deadline_hits,
        "deadline_misses": deadline_misses,
        "average_lateness_hours": avg_lateness,
        "provider_utilisation": prov_util,
    }


def print_report(metrics: Dict[str, Any]) -> None:
    """Pretty print metrics produced by :func:`evaluate`."""
    print("=== System Metrics ===")
    print(f"Makespan: {metrics['makespan_hours']:.2f} h")
    print(
        f"Throughput: {metrics['throughput_tasks_per_hour']:.2f} tasks/h\n"
        f"Deadline hits: {metrics['deadline_hits']}  misses: {metrics['deadline_misses']}\n"
        f"Average lateness: {metrics['average_lateness_hours']:.2f} h"
    )
    print("Provider utilisation:")
    for idx, util in metrics["provider_utilisation"].items():
        print(f"  Provider {idx}: {util:.2f}")
    print("Per-task summary:")
    for tid, rec in metrics["tasks"].items():
        start = rec["start"].isoformat() if rec["start"] else "-"
        finish = rec["finish"].isoformat() if rec["finish"] else "-"
        print(
            f"  Task {tid}: cost={rec['cost']:.2f}$ overrun={rec['budget_overrun']:.2f}$ "
            f"start={start} finish={finish} deadline_hit={rec['deadline_hit']} "
            f"lateness_h={rec['lateness_hours'] if rec['lateness_hours'] is not None else '-'}"
        )
