import itertools
from Core.Scheduler.interface import ComboGenerator

class BruteForceGenerator(ComboGenerator):
    def __init__(self, max_comb=81):
        self.max_comb = max_comb

    def best_combo(self, t, ps, now, ev, verbose=False):
        best = (float('-inf'), None)
        for cmb in itertools.islice(itertools.product(range(len(ps)), repeat=t.scene_number), self.max_comb):
            ok, t_tot, cost = ev.feasible(t, list(cmb), now, ps)
            if verbose:
                print(f"    → {cmb} ok={ok} t={t_tot:.2f}h cost={cost:.1f}$")
            if not ok:
                continue
            score = ev.efficiency(t, list(cmb), ps)
            if score > best[0]:
                best = (score, (list(cmb), t_tot, cost))
        return None if best[1] is None else best[1]

