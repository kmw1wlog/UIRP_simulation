import json
import datetime
from typing import Dict, List, Any, Tuple

from providers import providers
from tasks import tasks



class Scheduler:
    """스케줄링 로직 (하드코딩 전략 + 가중치 인자)"""
    def __init__(self, strategy: str = "earliest_deadline_first", optimization_constant: float = 1.0):
        self.strategy = strategy
        self.optimization_constant = optimization_constant
    # ---------------------- Core Algorithm -------------------------------
    def schedule_from_task(
        self, tasks: Tasks, providers: Providers
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Task 들을 Provider 에 배분하고, Provider index -> 할당리스트 반환"""
        # 1) 작업을 데드라인 기준 ascending 정렬
        task_list = sorted(tasks, key=lambda t: t.deadline)
        # 2) Provider 를 비용 기준으로 정렬
        sorted_providers = sorted(
            enumerate(providers.providers), key=lambda kv: kv[1].price_per_gpu_hour
        )
        for task in task_list:
            # 단순 모델: 1GPU 로 수행한다고 가정 → duration = workload / throughput
            duration_h = task.workload / sorted_providers[0][1].throughput
            assigned = False
            for idx, prov in sorted_providers:
                start = prov.earliest_available(duration_h)
                if start is None:
                    continue
                finish = start + datetime.timedelta(hours=duration_h)
                cost = duration_h * prov.price_per_gpu_hour
                if finish <= task.deadline and cost <= task.budget:
                    prov.assign(task.properties["id"], start, duration_h)
                    assigned = True
                    break
            # 조건 불만족 시 첫 번째 Provider 에 강제 배정
            if not assigned:
                idx, prov = sorted_providers[0]
                start = (
                    prov.earliest_available(duration_h)
                    or datetime.datetime.fromisoformat(prov.available_hours[0][0])
                )
                prov.assign(task.properties["id"], start, duration_h)
        # 3) 결과 매핑 생성
        out: Dict[int, List[Dict[str, Any]]] = {}
        for idx, prov in enumerate(providers.providers):
            out[idx] = [
                {
                    "task_id": tid,
                    "start": s.isoformat(),
                    "finish": f.isoformat(),
                }
                for tid, s, f in prov.schedule
            ]
        return out
class Simulator:
    """전체 시뮬레이션 래퍼"""
    def __init__(
        self,
        initialize_json_path: str,
        scheduler_strategy: str = "earliest_deadline_first",
        optimization_constant: float = 1.0,
    ) -> None:
        self.initialize_json_path = initialize_json_path
        with open(initialize_json_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        # --- Objects ------------------------------------------------------
        self.tasks = Tasks()
        self.tasks.initialize_from_data(config.get("tasks", []))
        self.providers = Providers()
        self.providers.initialize_from_data(config.get("providers", []))
        self.scheduler = Scheduler(scheduler_strategy, optimization_constant)
    # ------------------------ Simulation ---------------------------------
    def simulate(self) -> Dict[str, Any]:
        schedule_map = self.scheduler.schedule_from_task(self.tasks, self.providers)
        # --- KPI 계산 -----------------------------------------------------
        total_gpu_hours_used = 0.0
        total_gpu_hours_avail = 0.0
        for prov in self.providers.providers:
            # 사용시간
            for _, s, f in prov.schedule:
                total_gpu_hours_used += (f - s).total_seconds() / 3600.0
            # 가능시간
            for s_iso, f_iso in prov.available_hours:
                s = datetime.datetime.fromisoformat(s_iso)
                f = datetime.datetime.fromisoformat(f_iso)
                total_gpu_hours_avail += (f - s).total_seconds() / 3600.0
        idle_ratio = (
            1.0 - total_gpu_hours_used / total_gpu_hours_avail
            if total_gpu_hours_avail > 0
            else 0.0
        )
        task_metrics: List[Dict[str, Any]] = []
        for task in self.tasks:
            tid = task.properties["id"]
            record = next(
                (
                    (idx, s, f, prov)
                    for idx, prov in enumerate(self.providers.providers)
                    for (t_id, s, f) in prov.schedule
                    if t_id == tid
                ),
                None,
            )
            if record is None:
                task_metrics.append({"id": tid, "status": "UNASSIGNED"})
                continue
            idx, s, f, prov = record
            cost = (f - s).total_seconds() / 3600.0 * prov.price_per_gpu_hour
            task_metrics.append(
                {
                    "id": tid,
                    "provider_index": idx,
                    "start": s.isoformat(),
                    "finish": f.isoformat(),
                    "cost": cost,
                    "deadline_met": f <= task.deadline,
                    "budget_met": cost <= task.budget,
                }
            )
        return {
            "task_metrics": task_metrics,
            "idle_gpu_ratio": idle_ratio,
            "provider_schedules": schedule_map,
        }
if __name__ == "__main__":
    # 예시 실행. initialize.json 은 같은 디렉터리에 있다고 가정
    sim = Simulator("initialize.json")
    results = sim.simulate()
    print(json.dumps(results, indent=2, ensure_ascii=False))









