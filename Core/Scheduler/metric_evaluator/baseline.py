# Core/Scheduler/metric_evaluator/baseline.py
import math
import datetime
from typing import Dict, List
from Core.Scheduler.interface import MetricEvaluator
from Core.objective import calc_objective, DEFAULT_WEIGHTS

class BaselineEvaluator(MetricEvaluator):
    def __init__(self):
        self._c: Dict[tuple, float] = {}  # cache

    # ---- cached primitive times ------------------------------------------------
    def _t(self, t, s, p):
        k = ("tx", t.id, s, p)
        if k in self._c:
            return self._c[k]
        bw = min(t.bandwidth, p.bandwidth)
        v = float("inf") if bw <= 0 else (t.global_file_size + t.scene_size(s)) / bw / 3600
        self._c[k] = v
        return v

    def _cpt(self, t, p):
        k = ("cmp", t.id, p)
        if k in self._c:
            return self._c[k]
        v = float("inf") if getattr(p, "throughput", 0.0) <= 0 else t.scene_workload / p.throughput
        self._c[k] = v
        return v

    # ---- public API ------------------------------------------------------------
    def time_cost(self, t, s, p):
        # 이미 배정된 씬은 추가 시간/비용 0
        if t.scene_allocation_data[s][0] is not None:
            return 0.0, 0.0
        d = self._t(t, s, p) + self._cpt(t, p)
        return d, d * p.price_per_gpu_hour

    def _ea(self, prov, dur, cur):
        """earliest_available adaptor (1-arg or 2-arg)."""
        try:
            return prov.earliest_available(dur, cur)
        except TypeError:
            return prov.earliest_available(dur)

    def feasible(self, t, cmb, now, ps):
        # 선택된(= -1 제외) 씬만 provider별로 묶기
        grouped: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            if t.scene_allocation_data[sid][0] is not None:
                continue
            if p == -1:
                continue
            grouped.setdefault(p, []).append(sid)

        # 이미 사용한 비용(부분 배정 고려)
        spent = 0.0
        for prov in ps:
            for rec in getattr(prov, "schedule", []):
                if len(rec) == 3:
                    tid, st, ft = rec
                else:
                    tid, _, st, ft = rec
                if tid == t.id:
                    spent += ((ft - st).total_seconds() / 3600.0) * prov.price_per_gpu_hour

        end_latest = now
        incr_cost = 0.0

        # 실제 디스패치와 동일하게 provider-wise 연쇄 배치 시뮬레이션
        for p, sids in grouped.items():
            prov = ps[p]
            cur = now
            for sid in sids:
                dur, c = self.time_cost(t, sid, prov)
                if not math.isfinite(dur):
                    return False, math.inf, math.inf
                st = self._ea(prov, dur, cur)
                if st is None:
                    return False, math.inf, math.inf
                ft = st + datetime.timedelta(hours=dur)
                cur = ft
                incr_cost += c
            end_latest = max(end_latest, cur)

        if spent + incr_cost > t.budget:
            return False, (end_latest - now).total_seconds() / 3600.0, spent + incr_cost
        if end_latest > t.deadline:
            return False, (end_latest - now).total_seconds() / 3600.0, spent + incr_cost

        return True, (end_latest - now).total_seconds() / 3600.0, spent + incr_cost

    def efficiency(self, t, cmb: List[int], ps, now: datetime.datetime) -> float:
        total = 0.0
        # 배치된 씬에 대해서만 목적함수 합산
        for sid, pid in enumerate(cmb):
            if t.scene_allocation_data[sid][0] is not None:
                continue
            if pid == -1:
                continue
            prov = ps[pid]
            score = calc_objective(t, sid, prov)
            if score == float("inf") or score != score:  # NaN 방지
                return float("-inf")
            total += score
        # objective(작을수록 좋음)의 음수화 → 값이 클수록 더 좋은 조합
        return -total
