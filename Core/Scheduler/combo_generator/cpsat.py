# Core/Scheduler/combo_generator/cpsat.py
from __future__ import annotations
import math
import datetime as dt
from typing import List, Tuple
from ortools.sat.python import cp_model

from Core.Scheduler.interface import ComboGenerator

_SCALE = 1000
_BIG   = 10**9

# weights: (time, budget_penalty, deadline_penalty, provider_price, idle)
DEFAULT_WEIGHTS = (1.0, 200.0, 500.0, 1.0, 0.0)


def _cap_now_hours_from_avail(prov, now: dt.datetime) -> float:
    """Length of current available window starting at now (hours)."""
    for s, e in getattr(prov, "available_hours", []):
        if s <= now < e:
            return max(0.0, (e - now).total_seconds() / 3600.0)
    return 0.0


def _build_common_model(t, ps, now):
    S, P = t.scene_number, len(ps)
    TOT  = [[float("inf")]*P for _ in range(S)]
    COST = [[float("inf")]*P for _ in range(S)]
    PROF = [[0.0]*P for _ in range(S)]

    cap_hours = [_cap_now_hours_from_avail(prov, now) for prov in ps]

    # per-provider / per-scene transmission & compute times
    tx_global = [float("inf")] * P
    tx_scene  = [[float("inf")] * P for _ in range(S)]
    cmp_scene = [[float("inf")] * P for _ in range(S)]

    scene_work = getattr(t, "scene_workload", None)
    for p, prov in enumerate(ps):
        bw  = min(t.bandwidth, prov.bandwidth)
        thr = getattr(prov, "throughput", 0.0)
        if bw <= 0 or thr <= 0:
            continue
        tx_global[p] = (t.global_file_size / bw) / 3600.0
        for s in range(S):
            tx_scene[s][p] = (t.scene_size(s) / bw) / 3600.0
            if callable(scene_work):
                sw = scene_work(s)
            else:
                sw = scene_work if scene_work is not None else 0.0
            cmp_scene[s][p] = sw / thr

    # already used providers don't pay global overhead again
    used_providers = {t.scene_allocation_data[s][1]
                      for s in range(S)
                      if t.scene_allocation_data[s][0] is not None}
    for p in used_providers:
        tx_global[p] = 0.0

    # build per-scene metrics without global overhead
    for s in range(S):
        for p in range(P):
            prov = ps[p]
            tot_scene = tx_scene[s][p] + cmp_scene[s][p]
            total_time = tx_global[p] + tot_scene
            if total_time - 1e-9 > cap_hours[p]:
                tot_scene = float("inf")
            TOT[s][p] = tot_scene
            if math.isfinite(tot_scene):
                COST[s][p] = tot_scene * prov.price_per_gpu_hour
            PROF[s][p] = prov.price_per_gpu_hour

    # 이미 사용한 비용(부분 배정)
    spent = 0.0
    for prov in ps:
        for rec in getattr(prov, "schedule", []):
            if len(rec) == 3:
                tid, st, ft = rec
            else:
                tid, _, st, ft = rec
            if tid == t.id:
                spent += ((ft - st).total_seconds()/3600.0) * prov.price_per_gpu_hour
    remaining_budget = max(0.0, t.budget - spent)
    window_sec = int((t.deadline - now).total_seconds() * _SCALE)

    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{s}_{p}") for p in range(P)] for s in range(S)]
    y = [m.NewBoolVar(f"y{s}") for s in range(S)]  # 이번 스텝 배치 여부
    z = [m.NewBoolVar(f"z{p}") for p in range(P)]  # provider 사용 여부

    # 한 씬: 0개(미배치) 또는 1개 provider
    for s in range(S):
        m.Add(sum(x[s][p] for p in range(P)) == y[s])

    # provider당 1개 씬 제한
    for p in range(P):
        m.Add(sum(x[s][p] for s in range(S)) <= 1)

    # z와 x 연결
    for p in range(P):
        for s in range(S):
            m.Add(x[s][p] <= z[p])
        m.Add(sum(x[s][p] for s in range(S)) >= z[p])

    # 불가능 금지 + 이미 배정된 씬 고정
    for s in range(S):
        assigned = (t.scene_allocation_data[s][0] is not None)
        fixed_p = t.scene_allocation_data[s][1] if assigned else None
        for p in range(P):
            if not math.isfinite(TOT[s][p]):
                m.Add(x[s][p] == 0)
        if assigned:
            m.Add(y[s] == 1)
            for p in range(P):
                m.Add(x[s][p] == (1 if p == fixed_p else 0))
            for p in range(P):
                if p == fixed_p:
                    TOT[s][p] = COST[s][p] = 0.0
                    PROF[s][p] = 0.0
                    tx_scene[s][p] = cmp_scene[s][p] = 0.0
                else:
                    TOT[s][p] = COST[s][p] = float("inf")
                    tx_scene[s][p] = cmp_scene[s][p] = float("inf")
            m.Add(z[fixed_p] == 1)

    # 정수 스케일링
    tot_int  = [[int(TOT[s][p]*3600*_SCALE) if math.isfinite(TOT[s][p]) else _BIG for p in range(P)] for s in range(S)]
    cost_int = [[int(COST[s][p]*_SCALE)     if math.isfinite(COST[s][p]) else _BIG for p in range(P)] for s in range(S)]
    prof_int = [[int(PROF[s][p]*_SCALE)     for p in range(P)] for s in range(S)]
    txg_int  = [int(tx_global[p]*3600*_SCALE) if math.isfinite(tx_global[p]) else _BIG for p in range(P)]
    txs_int  = [[int(tx_scene[s][p]*3600*_SCALE) if math.isfinite(tx_scene[s][p]) else _BIG for p in range(P)] for s in range(S)]
    cmps_int = [[int(cmp_scene[s][p]*3600*_SCALE) if math.isfinite(cmp_scene[s][p]) else _BIG for p in range(P)] for s in range(S)]
    price_int = [int(ps[p].price_per_gpu_hour * _SCALE) for p in range(P)]

    prov_time = [m.NewIntVar(0, _BIG, f"time_p{p}") for p in range(P)]
    for p in range(P):
        m.Add(prov_time[p] ==
              txg_int[p] * z[p] +
              sum((txs_int[s][p] + cmps_int[s][p]) * x[s][p] for s in range(S)))
    makespan = m.NewIntVar(0, _BIG, "makespan")
    m.AddMaxEquality(makespan, prov_time)

    total_cost = m.NewIntVar(0, _BIG, "total_cost")
    m.Add(total_cost ==
          sum((txg_int[p] * z[p] +
               sum((txs_int[s][p] + cmps_int[s][p]) * x[s][p] for s in range(S))) * price_int[p]
              for p in range(P)))
    over_budget = m.NewIntVar(0, _BIG, "over_budget")
    m.Add(over_budget >= total_cost - int(remaining_budget * _SCALE))
    over_deadline = m.NewIntVar(0, _BIG, "over_deadline")
    m.Add(over_deadline >= makespan - window_sec)

    # 가능하면 최소 1개 이상 배치
    unassigned = [s for s in range(S) if t.scene_allocation_data[s][0] is None]
    has_any_finite = any(tot_int[s][p] < _BIG for s in unassigned for p in range(P))
    if has_any_finite:
        m.Add(sum(y[s] for s in unassigned) >= 1)

    return m, x, y, tot_int, cost_int, prof_int, total_cost, makespan, over_budget, over_deadline

class CPSatComboGenerator(ComboGenerator):
    def time_complexity(self, t, ps, now, ev):
        unassigned = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        feasible = []
        for sid in unassigned:
            cands = []
            for p_idx, prov in enumerate(ps):
                d, _ = ev.time_cost(t, sid, prov)
                if math.isfinite(d) and d > 0 and _cap_now_hours_from_avail(prov, now) >= d:
                    cands.append(p_idx)
            feasible.append(cands)

        from functools import lru_cache

        @lru_cache(None)
        def dfs(i, mask):
            if i == len(feasible):
                return 1
            total = dfs(i + 1, mask)  # skip
            for p in feasible[i]:
                if mask & (1 << p) == 0:
                    total += dfs(i + 1, mask | (1 << p))
            return total

        return dfs(0, 0) - 1
    def best_combo(self, t, ps, now, ev, verbose=False):
        if verbose:
            space = self.time_complexity(t, ps, now, ev)
            print(f"[CP] search space={space}")
        a1, a2, a3, b1, _ = DEFAULT_WEIGHTS

        # 공통 제약 구성
        m1, x1, y1, *_ = _build_common_model(t, ps, now)

        # 1단계: 배치 수 최대화
        m1.Maximize(sum(y1[s] for s in range(t.scene_number)
                        if t.scene_allocation_data[s][0] is None))
        solver1 = cp_model.CpSolver()
        if verbose:
            solver1.parameters.log_search_progress = True
        solver1.parameters.max_time_in_seconds = 10
        status1 = solver1.Solve(m1)
        if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        y_opt = int(sum(solver1.Value(y1[s]) for s in range(t.scene_number)
                        if t.scene_allocation_data[s][0] is None))
        if y_opt == 0:
            return None  # 이번 스텝 배치 불가

        # 2단계: 배치 수를 y_opt 이상으로 고정하고, 시간/비용 패널티 최소화
        m2, x2, y2, tot_int2, cost_int2, prof_int2, total_cost2, makespan2, overB2, overDL2 = _build_common_model(t, ps, now)
        m2.Add(sum(y2[s] for s in range(t.scene_number)
                   if t.scene_allocation_data[s][0] is None) >= y_opt)

        prof_sum = sum(prof_int2[s][p] * x2[s][p] for s in range(t.scene_number) for p in range(len(ps)))
        m2.Minimize(int(a1*_SCALE) * makespan2 +
                    int(a2*_SCALE) * overB2 +
                    int(a3*_SCALE) * overDL2 +
                    int(b1*_SCALE) * prof_sum)

        solver2 = cp_model.CpSolver()
        if verbose:
            solver2.parameters.log_search_progress = True
        solver2.parameters.max_time_in_seconds = 10
        status2 = solver2.Solve(m2)
        if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        # 해 추출
        cmb: List[int] = []
        for s in range(t.scene_number):
            if t.scene_allocation_data[s][0] is not None:
                cmb.append(t.scene_allocation_data[s][1])
                continue
            if solver2.BooleanValue(y2[s]) == 0:
                cmb.append(-1)
                continue
            chosen = None
            for p in range(len(ps)):
                if solver2.BooleanValue(x2[s][p]):
                    chosen = p
                    break
            cmb.append(chosen if chosen is not None else -1)

        ok, t_tot, cost, _, _, _ = ev.feasible(t, cmb, now, ps)
        if not ok:
            return None
        return cmb, t_tot, cost
