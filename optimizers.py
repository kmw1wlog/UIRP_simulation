# optimizers.py  (추가/수정 부분만 발췌)

from abc import ABC, abstractmethod
import datetime
from typing import List, Tuple
from tasks import Task
from providers import Providers, Provider
from objective import calc_objective, DEFAULT_WEIGHTS

# 타입 별칭
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

# ───────────────────────────────────────────────
# ❶ Scene‑level Optimizer 인터페이스
# ───────────────────────────────────────────────
class BaseSceneOptimizer(ABC):
    @abstractmethod
    def select(
        self,
        task: Task,
        scene_id: int,
        now: datetime.datetime,
        providers: Providers,
        weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS,
    ) -> int | None:          # provider index or None
        """대상 scene 에 대해 할당할 provider idx 반환 (없으면 None)"""


# ───────────────────────────────────────────────
# ❷ 그리디 구현: feasible provider 중 objective 최소
# ───────────────────────────────────────────────
class GreedySceneOptimizer(BaseSceneOptimizer):
    def select(
        self,
        task: Task,
        scene_id: int,
        now: datetime.datetime,
        providers: Providers,
        weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS,
    ) -> int | None:
        scene_size = task.scene_size(scene_id)
        best_idx, best_obj = None, float("inf")

        for idx, prov in enumerate(providers):
            # 1) 전송 + 연산 시간 계산
            rate = min(task.bandwidth, prov.bandwidth)
            tx_h  = (task.global_file_size + scene_size) / rate
            cmp_h = task.scene_workload / prov.throughput
            earliest = prov.earliest_available(cmp_h, after=now + datetime.timedelta(hours=tx_h))
            if earliest is None:          # 해당 provider 는 당장 수행 불가
                continue

            # 2) objective 평가
            obj = calc_objective(task, scene_id, prov, weights)
            if obj < best_obj:
                best_obj, best_idx = obj, idx

        return best_idx
