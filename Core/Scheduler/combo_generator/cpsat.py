try:
    from ortools.sat.python import cp_model
    import math
    from Core.Scheduler.interface import ComboGenerator

    class CPSatComboGenerator(ComboGenerator):
        def best_combo(self, t, ps, now, ev, verbose=False):
            m = cp_model.CpModel()
            S, P = t.scene_number, len(ps)
            x = [[m.NewBoolVar(f"x{s}_{p}") for p in range(P)] for s in range(S)]
            for s in range(S):
                m.Add(sum(x[s]) == 1)

            big = 10**7
            objective_terms = []
            for s in range(S):
                for p in range(P):
                    obj = ev.efficiency(t, [p if i == s else 0 for i in range(S)], ps)  # crude 1-hot eval
                    score = int(-obj * 1000) if math.isfinite(obj) else big
                    objective_terms.append(score * x[s][p])

            m.Minimize(sum(objective_terms))

            slv = cp_model.CpSolver()
            if verbose:
                slv.parameters.log_search_progress = True
            slv.parameters.max_time_in_seconds = 5

            st = slv.Solve(m)
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None
            cmb = [next(p for p in range(P) if slv.BooleanValue(x[s][p])) for s in range(S)]
            return cmb, 0.0, 0.0  # t_tot, cost ignored here

except ImportError:
    CPSatComboGenerator = None
