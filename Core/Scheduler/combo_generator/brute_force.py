# Core/Scheduler/combo_generator/brute_force.py
import itertools, math
from itertools import combinations, product
from Core.Scheduler.interface import ComboGenerator

class BruteForceGenerator(ComboGenerator):
    def __init__(self, max_comb=None):
        self.max_comb = max_comb  # 사용 안 함(탐색 축소는 top-k로 처리)

    def _best_providers(self, t, ps, sid, ev, kprov=3):
        cand = []
        for p_idx, prov in enumerate(ps):
            d, _ = ev.time_cost(t, sid, prov)
            if math.isfinite(d):
                cand.append((d, p_idx))
        cand.sort(key=lambda x: x[0])
        return [p for _, p in cand[:max(1, min(kprov, len(cand)))]]

    def best_combo(self, t, ps, now, ev, verbose=False):
        # 아직 미배정인 씬만 대상
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        # 씬별 후보 provider 목록(짧은 시간 상위 k)
        kprov = min(3, len(ps))
        prov_cands = {sid: self._best_providers(t, ps, sid, ev, kprov=kprov) for sid in scene_ids}

        # 1) 가장 많은 씬을 배치하는 해를 먼저 찾는다 (k = n..1)
        n = len(scene_ids)
        for k in range(n, 0, -1):
            best = (float("-inf"), None)  # (score, (cmb, t_tot, cost))
            for subset in combinations(scene_ids, k):
                # subset 아닌 씬은 -1(이번 스텝 미배치)
                # subset인 씬은 provider 후보 중에서 product
                for picks in product(*(prov_cands[sid] for sid in subset)):
                    # cmb 구성
                    cmb = [-1] * t.scene_number
                    for sid, pid in zip(subset, picks):
                        cmb[sid] = pid

                    ok, t_tot, cost = ev.feasible(t, cmb, now, ps)
                    if not ok:
                        continue
                    score = ev.efficiency(t, cmb, ps, now)
                    if score > best[0]:
                        best = (score, (cmb, t_tot, cost))

            if best[1] is not None:
                # k개 배치 가능한 해를 찾았으므로 즉시 반환 (배치 수 최대 보장)
                if verbose:
                    print(f"[BF] place {k}/{n} scenes this step")
                return best[1]

        # 어떤 씬도 이번 스텝에 배치 불가
        return None
