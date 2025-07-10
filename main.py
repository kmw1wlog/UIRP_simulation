import json
import datetime
from typing import Dict, List, Any, Tuple


class Task:

    def __init__(self, properties: Dict[str, Any]):
        self.properties = properties
        # === Local variables ===
        self.workload: float = properties.get("workload", 0.0)
        self.bandwidth: float = properties.get("bandwidth", 0.0)
        self.budget: float = properties.get("budget", float("inf"))
        dl = properties.get("deadline")
        self.deadline: datetime.datetime = (
            datetime.datetime.fromisoformat(dl) if isinstance(dl, str) else dl
        )

class Tasks:
    """Task 컨테이너 (dict)"""
    def __init__(self):
        # Local variable: {task_id: Task}
        self.tasks: Dict[str, Task] = {}
    # -------------------- Public API -------------------------------------
    def initialize_from_data(self, data: List[Dict[str, Any]]) -> None:
        """리스트 형태의 원시 dict 로부터 Task 객체 생성"""
        for item in data:
            tid = item.get("id")
            if tid is None:
                raise ValueError("Task item must have an 'id' field")
            self.tasks[tid] = Task(item)
    # -------------------- Convenience ------------------------------------
    def __iter__(self):
        return iter(self.tasks.values())

    def __getitem__(self, item):
        return self.tasks[item]

class Provider:
    """GPU 제공자 노드"""
    def __init__(self, properties: Dict[str, Any]):
        self.properties = properties
        # === Local variables ===
        self.throughput: float = properties.get("throughput", 1.0)  # (GPU‑hour processed)/hour/GPU
        self.available_hours: List[Tuple[str, str]] = properties.get("available_hours", [])
        self.price_per_gpu_hour: float = properties.get("price", 0.0)
        self.bandwidth: float = properties.get("bandwidth", 0.0)
        # 스케줄: (task_id, start, finish)
        self.schedule: List[Tuple[str, datetime.datetime, datetime.datetime]] = []
    # ------------------------ Scheduling Helpers -------------------------
    def earliest_available(self, duration_h: float) -> datetime.datetime | None:
        """duration_h 만큼 연속으로 작업 가능한 가장 빠른 시작 시각 반환"""
        intervals = [
            (
                datetime.datetime.fromisoformat(s),
                datetime.datetime.fromisoformat(e),
            )
            for s, e in self.available_hours
        ]
        intervals.sort(key=lambda t: t[0])

        # (t1, t2 , t3  ... )  (true, false, true ... )
        t_list = [datetime.datetime.now()]
        avail_list = [False]

        for start, end in intervals:
            t_list.append(start)
            t_list.append(end)
            avail_list.append(True)
            avail_list.append(False)

        for index, (t, avail) in enumerate(zip(t_list, avail_list)):
            if avail:
                print("여유시간: ", t_list[index+1] - t)
                if (t_list[index + 1] - t).total_seconds() / 3600.0 > duration_h:
                    return





            # 가능할떄만 current 던지기
        return None  # 슬롯 없음
    def assign(self, task_id: str, start: datetime.datetime, duration_h: float) -> None:
        finish = start + datetime.timedelta(hours=duration_h)
        self.schedule.append((task_id, start, finish))
class Providers:
    """Provider 컨테이너 (list)"""
    def __init__(self):
        self.providers: List[Provider] = []

    def initialize_from_data(self, data: List[Dict[str, Any]]) -> None:
        for item in data:
            self.providers.append(Provider(item))

    def __iter__(self):
        return iter(self.providers)

    def __getitem__(self, item):
        return self.providers[item]

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









