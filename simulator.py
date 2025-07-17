
from __future__ import annotations
import json, datetime, sys, pprint
from pathlib import Path
from typing import List, Tuple, Dict, Any

from tasks import Tasks, Task
from providers import Providers
from scheduler import Scheduler, Assignment
from objective import calc_objective
from utils import merge_intervals


class Simulator:
    """Wrapper = load → schedule → evaluate."""

    # -------------------------------------------------------------
    # construction
    # -------------------------------------------------------------
    def __init__(self, config_path: str):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

        # objects
        self.tasks = Tasks()
        self.tasks.initialize_from_data(cfg.get("tasks", []))

        self.providers = Providers()
        self.providers.initialize_from_data(cfg.get("providers", []))

        self.results: List[Assignment] = []

    # -------------------------------------------------------------
    # scheduling
    # -------------------------------------------------------------
    def schedule(self, scheduler: Scheduler):
        """Run the supplied scheduler and store Allocation records."""
        self.results = scheduler.run(self.tasks, self.providers)

    # -------------------------------------------------------------
    # evaluation helpers
    # -------------------------------------------------------------
    @staticmethod
    def _missing_scenes(task: Task) -> List[int]:
        return [
            idx for idx, (st, _) in enumerate(task.scene_allocation_data) if st is None
        ]

    def _provider_idle_ratio(self) -> float:
        """Weighted mean idle ratio across all providers."""
        if not self.results:
            return 1.0

        # horizon across *all* allocations
        all_s = min(r[2] for r in self.results)
        all_f = max(r[3] for r in self.results)
        horizon_h = (all_f - all_s).total_seconds() / 3600.0
        if horizon_h <= 0:
            return 1.0

        # busy time per provider (union of intervals)
        prov_busy_h: Dict[int, float] = {i: 0.0 for i in range(len(self.providers))}
        for idx, prov in enumerate(self.providers):
            intervals = [(s, f) for _, _, s, f in prov.schedule]
            busy = merge_intervals(intervals)
            prov_busy_h[idx] = sum(
                (e - s).total_seconds() / 3600.0 for s, e in busy
            )

        total_busy = sum(prov_busy_h.values())
        return max(0.0, 1.0 - total_busy / (horizon_h * len(self.providers)))

    # -------------------------------------------------------------
    # evaluation main
    # -------------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        if not self.results:
            raise RuntimeError("schedule() has not been run or produced no results.")

        task_stats: Dict[str, Any] = {}
        finished_records: List[Assignment] = []
        objective_sum = 0.0
        total_cost = 0.0

        # --- per-task metrics ------------------------------------
        for task in self.tasks:
            recs = [r for r in self.results if r[0] == task.id]
            missing = self._missing_scenes(task)

            task_cost = sum(
                (r[3] - r[2]).total_seconds() / 3600.0
                * self.providers[r[4]].price_per_gpu_hour
                for r in recs
            )
            total_cost += task_cost

            if missing:
                # incomplete
                task_stats[task.id] = {
                    "completed": False,
                    "missing_scenes": missing,
                    "cost": task_cost,
                    "budget_ok": task_cost <= task.budget,
                }
                continue

            # completed
            starts = [r[2] for r in recs]
            finishes = [r[3] for r in recs]
            completion_h = (max(finishes) - min(starts)).total_seconds() / 3600.0

            # objective sum (scene-level)
            for r in recs:
                objective_sum += calc_objective(
                    task, r[1], self.providers[r[4]]
                )

            task_stats[task.id] = {
                "completed": True,
                "completion_h": completion_h,
                "cost": task_cost,
                "budget_ok": task_cost <= task.budget,
                "deadline_ok": max(finishes) <= task.deadline,
            }
            finished_records.extend(recs)

        # --- global KPIs -----------------------------------------
        makespan_h = (
            (max(r[3] for r in finished_records) - min(r[2] for r in finished_records)
             ).total_seconds() / 3600.0
            if finished_records
            else 0.0
        )
        idle_ratio = self._provider_idle_ratio()

        return {
            "tasks": task_stats,
            "makespan_h": makespan_h,
            "overall_idle_ratio": idle_ratio,
            "total_cost": total_cost,
            "objective_sum": objective_sum,
        }


# -------------------------------------------------------------
# simple CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"

    sim = Simulator(cfg_path)
    sch = Scheduler()            # default weights & time_gap
    sim.schedule(sch)
    metrics = sim.evaluate()

    pprint.pprint(metrics, sort_dicts=False)