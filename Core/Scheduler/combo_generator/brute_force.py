# Core/Scheduler/combo_generator/brute_force.py
from __future__ import annotations
import math
from itertools import product
from Core.Scheduler.interface import ComboGenerator

class BruteForceGenerator(ComboGenerator):
    """
    한 타임스텝에서 'provider당 최대 1개 씬'을 하드 제약.
    씬별로 시간 짧은 상위 k 후보 + 연기(-1)를 조합해 탐색.
    """
    def __init__(self, kprov: int = 3):
        self.kprov = kprov

    def _best_providers(self, t, ps, sid, ev, kprov=3):
        cand = []
        for p_idx, prov in enumerate(ps):
            d, _ = ev.time_cost(t, sid, prov)
            if math.isfinite(d) and d > 0:
                cand.append((d, p_idx))
        cand.sort(key=lambda x: x[0])
        return [p for _, p in cand[:max(1, min(kprov, len(cand)))]]

    def best_combo(self, t, ps, now, ev, verbose=False):
        # 미배정 씬만
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        # 씬별 후보: 시간 짧은 상위 k + skip(-1)
        kprov = min(self.kprov, len(ps))
        cand_lists = []
        for sid in scene_ids:
            cands = self._best_providers(t, ps, sid, ev, kprov=kprov)
            cands.append(-1)  # 연기 옵션
            cand_lists.append((sid, cands))

        best = (float("-inf"), None)
        for picks in product(*[cl[1] for cl in cand_lists]):
            # 전량 연기 제외
            if not any(pid != -1 for pid in picks):
                continue
            # HARD: provider 중복 배정 금지
            used = [pid for pid in picks if pid != -1]
            if len(set(used)) != len(used):
                continue

            cmb = [-1] * t.scene_number
            for (sid, _), pid in zip(cand_lists, picks):
                cmb[sid] = pid

            ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
            if not ok:
                continue
            score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
            if score > best[0]:
                best = (score, (cmb, t_tot, cost))

        return None if best[1] is None else best[1]
