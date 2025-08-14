# Core/Scheduler/combo_generator/greedy.py
from __future__ import annotations
import heapq
from Core.Scheduler.interface import ComboGenerator


class GreedyComboGenerator(ComboGenerator):
    """Greedy heuristic for scene-provider assignment.

    Each unassigned scene is paired with every provider. The pairwise
    combinations are evaluated individually and pushed into a max-heap by
    efficiency. Pairs are popped from the heap and selected if the scene and
    provider have not been used yet. The final chosen combination is validated
    with ``evaluator.feasible`` and returned.
    """

    def time_complexity(self, t, ps, now, ev):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        return len(scene_ids) * len(ps)

    def best_combo(self, t, ps, now, ev, verbose=False):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        heap = []  # max-heap using negative score
        for sid in scene_ids:
            for pid, _ in enumerate(ps):
                cmb = [-1] * t.scene_number
                cmb[sid] = pid
                ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
                if not ok:
                    continue
                score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
                heapq.heappush(heap, (-score, sid, pid))

        cmb = [-1] * t.scene_number
        used = set()
        while heap:
            neg_score, sid, pid = heapq.heappop(heap)
            if cmb[sid] != -1 or pid in used:
                continue
            cmb[sid] = pid
            used.add(pid)

        if all(p == -1 for p in cmb):
            return None

        ok, t_tot, cost, _, _, _ = ev.feasible(t, cmb, now, ps)
        if not ok:
            return None
        return cmb, t_tot, cost
