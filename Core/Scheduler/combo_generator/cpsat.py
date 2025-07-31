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
            dur = [[0] * P for _ in range(S)]
            cost = [[0] * P for _ in range(S)]
            for s in range(S):
                for p in range(P):
                    d, c = ev.time_cost(t, s, ps[p])
                    dur[s][p] = big if math.isinf(d) else int(d * 3600)
                    cost[s][p] = big if math.isinf(c) else int(c * 100)

            makespan = m.NewIntVar(0, big, "mk")
            for p in range(P):
                tot = m.NewIntVar(0, big, f"tot{p}")
                m.Add(tot == sum(dur[s][p] * x[s][p] for s in range(S)))
                m.Add(tot <= makespan)

            total_cost = sum(cost[s][p] * x[s][p] for s in range(S) for p in range(P))
            m.Add(total_cost <= int(t.budget * 100))
            m.Add(makespan <= int((t.deadline - now).total_seconds()))

            m.Minimize(makespan)
            slv = cp_model.CpSolver()
            if verbose:
                slv.parameters.log_search_progress = True
            slv.parameters.max_time_in_seconds = 5

            st = slv.Solve(m)
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None
            cmb = [next(p for p in range(P) if slv.BooleanValue(x[s][p])) for s in range(S)]
            return cmb, slv.Value(makespan) / 3600, slv.Value(total_cost) / 100

except ImportError:
    CPSatComboGenerator = None
