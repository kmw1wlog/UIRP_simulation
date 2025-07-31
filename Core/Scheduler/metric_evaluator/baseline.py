import math
import datetime
from typing import Dict, List
from Core.Scheduler.interface import MetricEvaluator


class BaselineEvaluator(MetricEvaluator):
    def __init__(self):
        self._c: Dict[tuple, float] = {}

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
        v = float("inf") if p.throughput <= 0 else t.scene_workload / p.throughput
        self._c[k] = v
        return v

    def time_cost(self, t, s, p):
        d = self._t(t, s, p) + self._cpt(t, p)
        return d, d * p.price_per_gpu_hour

    def feasible(self, t, cmb, now, ps):
        grouped: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            grouped.setdefault(p, []).append(sid)
        t_tot, cost = 0.0, 0.0
        for p, sids in grouped.items():
            if any(math.isinf(self.time_cost(t, sid, ps[p])[0]) for sid in sids):
                return False, math.inf, math.inf
            seq = sum(self.time_cost(t, sid, ps[p])[0] for sid in sids)
            t_tot = max(t_tot, seq)
            cost += sum(self.time_cost(t, sid, ps[p])[1] for sid in sids)
        if cost > t.budget:
            return False, t_tot, cost
        if now + datetime.timedelta(hours=t_tot) > t.deadline:
            return False, t_tot, cost
        return True, t_tot, cost

    def efficiency(self, t, t_tot, cost):
        return 0.0 if t_tot <= 0 or cost <= 0 else (t.scene_number * t.scene_workload) / (cost * t_tot)