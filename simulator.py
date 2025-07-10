# simulation.py
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

from tasks import Tasks, Task
from providers import Providers, Provider
from scheduler import Scheduler


class Simulator:
    """
    1) JSON 로부터 Tasks, Providers 생성
    2) Scheduler (Earliest-Deadline-First, Provider별 throughput 사용)
    3) evaluate() : 정량 메트릭 반환
    """

    # ---------------- init ----------------
    def __init__(self, config_path: str):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

        # 객체 초기화
        self.tasks = Tasks()
        self.tasks.initialize_from_data(cfg.get("tasks", []))
        self.providers = Providers()
        self.providers.initialize_from_data(cfg.get("providers", []))

        # 결과: (task_id, scene_id, start, finish, provider_idx)
        self.results: List[Tuple[str, int, datetime.datetime, datetime.datetime, int]] = []


    # ------------- Scheduler --------------
    def schedule(self, scheduler: Scheduler):
        self.results = scheduler.run(self.tasks, self.providers)


    # --------------- Metrics --------------
    def _task_missing_scenes(self, task: Task) -> List[int]:
        return [
            idx for idx, (st, _) in enumerate(task.scene_allocation_data) if st is None
        ]

    # --------------- Metrics --------------
    def evaluate(self) -> Dict[str, Any]:
        """Task 가 미완료될 경우 completed=False · missing_scenes 리스트 추가"""
        if not self.results:           # 스케줄 실패 or 비어 있을 때
            raise RuntimeError("No scheduling result to evaluate")

        task_stats: Dict[str, Dict[str, Any]] = {}
        finished_records: List[Tuple] = []

        for task in self.tasks:
            recs = [r for r in self.results if r[0] == task.id]
            missing = self._task_missing_scenes(task)

            if missing:   # ★ 모든 scene 배정이 안 된 경우
                task_stats[task.id] = {
                    "completed": False,
                    "missing_scenes": missing,
                    "cost": sum(
                        (r[3] - r[2]).total_seconds() / 3600.0 * self.providers[r[4]].price_per_gpu_hour
                        for r in recs
                    ),
                }
                # 미완료 Task 는 makespan·idle 계산에서 제외
                continue

            # ---- 정상 완료 Task ----
            starts   = [r[2] for r in recs]
            finishes = [r[3] for r in recs]
            finished_records.extend(recs)

            completion_h = (max(finishes) - min(starts)).total_seconds() / 3600.0
            cost = sum(
                (r[3] - r[2]).total_seconds() / 3600.0 * self.providers[r[4]].price_per_gpu_hour
                for r in recs
            )

            task_stats[task.id] = {
                "completed": True,
                "completion_h": completion_h,
                "cost": cost,
                "budget_ok": cost <= task.budget,
                "deadline_ok": max(finishes) <= task.deadline,
            }

        # ---- 전체 메트릭 (완료된 Task 기준) ----
        if finished_records:
            all_start = min(r[2] for r in finished_records)
            all_end   = max(r[3] for r in finished_records)
            horizon_h = (all_end - all_start).total_seconds() / 3600.0

            busy_h = sum(
                (r[3] - r[2]).total_seconds() / 3600.0 for r in finished_records
            )
            idle_ratio = 1.0 - busy_h / (horizon_h * len(self.providers))
        else:
            # 아무 Task 도 완료되지 못한 극단 상황
            horizon_h = 0.0
            idle_ratio = 1.0

        return {
            "tasks": task_stats,
            "makespan_h": horizon_h,
            "overall_idle_ratio": idle_ratio,
        }


# ---------------- CLI ----------------
if __name__ == "__main__":
    import sys, pprint
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.json"

    sim = Simulator(cfg)
    sch = Scheduler()
    sim.schedule(sch)
    metrics = sim.evaluate()

    print(sim.results)

    pprint.pprint(metrics)
