# scheduler.py
import datetime
from typing import List, Tuple, Optional

from tasks import Tasks, Task
from providers import Providers, Provider

Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]
#              task_id  scene  start                        finish   prov_idx


class Scheduler:
    """
    간단 EDF(마감 기한 우선) 온라인 스케줄러.
    - 시계를 `time_gap` 간격으로 전진하며
    - now 이전에 도착한 씬을 대기 큐(EDF)로 넣고
    - 할당 가능하면 즉시 Provider 에 배정
    """

    # ---------------- init ----------------
    def __init__(
        self,
        time_gap: datetime.timedelta = datetime.timedelta(minutes=30),
    ):
        self.time_gap = time_gap
        self.waiting_scenes: List[Tuple[Task, int]] = []       # (task, scene_id)
        self.results: List[Assignment] = []

    # ------------------ helper ------------------
    @staticmethod
    def _scene_key(task: Task, scene_id: int) -> Tuple[str, int]:
        """큐 중복 검사를 위한 (task_id, scene_id) 키"""
        return (task.id, scene_id)

    # -------- 이벤트 주입: Task arrival ----------
    def _feed(self, now: datetime.datetime, tasks: Tasks) -> None:
        """
        start_time ≤ now 인 scene 을 waiting 큐에 넣는다.
        이미 큐에 있거나 배정된 scene 은 건너뜀.
        """
        existing_keys = {self._scene_key(t, s) for t, s in self.waiting_scenes}

        for task in tasks:
            if task.start_time <= now:
                for s in range(task.scene_number):
                    if task.scene_allocation_data[s][0] is not None:
                        continue                # 이미 배정
                    key = self._scene_key(task, s)
                    if key not in existing_keys:
                        self.waiting_scenes.append((task, s))
                        existing_keys.add(key)

        # EDF 정렬
        self.waiting_scenes.sort(key=lambda ts: ts[0].deadline)

    # ---------------- 스케줄 한 스텝 ----------------
    def _schedule_once(
        self,
        now: datetime.datetime,
        providers: Providers,
    ) -> List[Assignment]:
        """
        waiting 큐를 순회하며 할당 가능한 씬을 바로 배정
        (충돌 및 중복은 여기서 최종 확인)
        """
        new_assignments: List[Assignment] = []
        still_waiting: List[Tuple[Task, int]] = []

        for task, scene_id in self.waiting_scenes:
            # 직전에 다른 스레드(루프)에서 배정됐을 수도 있으니 재확인
            if task.scene_allocation_data[scene_id][0] is not None:
                continue

            # Provider 후보 탐색
            best_choice: Optional[Tuple[int, float, datetime.datetime]] = None
            for idx, prov in enumerate(providers):
                dur_h = task.scene_workload / prov.throughput
                start = prov.earliest_available(dur_h)  # Provider 내부 충돌 체크
                if start and start <= now:
                    if (best_choice is None or
                        prov.price_per_gpu_hour < providers[best_choice[0]].price_per_gpu_hour):
                        best_choice = (idx, dur_h, start)

            if best_choice:
                idx, dur_h, start = best_choice
                prov = providers[idx]
                # 최종 중복 방지: assign 직전 Task 상태 다시 확인
                if task.scene_allocation_data[scene_id][0] is None:
                    prov.assign(task.id, scene_id, start, dur_h)
                    finish = start + datetime.timedelta(hours=dur_h)
                    task.scene_allocation_data[scene_id] = (start, idx)
                    new_assignments.append((task.id, scene_id, start, finish, idx))
            else:
                still_waiting.append((task, scene_id))

        self.waiting_scenes = still_waiting
        return new_assignments

    # ------------------- run -------------------
    def run(
        self,
        tasks: Tasks,
        providers: Providers,
        time_start: Optional[datetime.datetime] = None,
        time_end: Optional[datetime.datetime] = None,
    ) -> List[Assignment]:
        """
        시뮬레이션 메인 루프.
        - time_start 미지정 시 earliest(Task.start, Provider.available)
        - time_end   미지정 시 latest(deadline) + 1 day
        """
        if time_start is None:
            task_min = min(t.start_time for t in tasks)
            prov_min = min(p.available_hours[0][0] for p in providers)
            time_start = min(task_min, prov_min)

        if time_end is None:
            time_end = max(t.deadline for t in tasks) + datetime.timedelta(days=1)

        now = time_start
        while now < time_end:
            self._feed(now, tasks)
            self.results.extend(self._schedule_once(now, providers))
            now += self.time_gap

            # 모든 scene 이 배정되면 조기 종료
            if all(all(st is not None for st, _ in t.scene_allocation_data) for t in tasks):
                break

        return self.results
