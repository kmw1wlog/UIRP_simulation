# run_experiments.py  –  최종 안정판
import importlib, json, tempfile, argparse, sys
from pathlib import Path
from statistics import mean

gen_config = importlib.import_module("gen_config")
Scheduler  = importlib.import_module("scheduler").Scheduler
Simulator  = importlib.import_module("simulator").Simulator

def run_batch(runs: int, n_tasks: int, n_prov: int,
              seed: int, details: bool) -> None:
    mk_, idle_, comp_, cost_, obj_ = [], [], [], [], []
    skipped = 0

    for i in range(runs):
        cfg = Path(tempfile.gettempdir()) / f"cfg_{i}.json"
        gen_config.generate_cfg(n_tasks, n_prov, seed + i, str(cfg))

        sim = Simulator(str(cfg))
        sim.schedule(Scheduler())

        # ---------- 스케줄 결과가 없으면 evaluate() 호출 X ----------
        if not sim.results:
            print(f"[{i+1:02}/{runs}] ⚠ 스케줄 할당 0건 → skip")
            skipped += 1
            print("-" * 60)
            continue

        try:
            m = sim.evaluate()
        except RuntimeError as e:
            print(f"[{i+1:02}/{runs}] ⚠ evaluate 오류: {e} → skip")
            skipped += 1
            print("-" * 60)
            continue

        # ---------- 메트릭 수집 ----------
        mk_.append(mk := m["makespan_h"])
        idle_.append(idle := m["overall_idle_ratio"])
        comp_rate = sum(1 for v in m["tasks"].values() if v["completed"]) / len(m["tasks"])
        comp_.append(comp_rate)
        cost_.append(cost := m["total_cost"])
        obj_.append(obj := m["objective_sum"])

        hdr = (f"[{i+1:02}/{runs}] makespan {mk:.1f}h | idle {idle:.2%} | "
               f"complete {comp_rate:.0%} | cost {cost:.1f} | obj {obj:.1f}")
        print(hdr)

        if details:
            print("  ├─ schedule:")
            print(json.dumps(sim.results, default=str, indent=2, ensure_ascii=False))
            print("  ╰─ metrics:")
            print(json.dumps(m, default=str, indent=2, ensure_ascii=False))
            print("-" * 80)

    # ---------- 요약 ----------
    done = runs - skipped
    print("\n=== Summary ===")
    print(f"runs attempted       : {runs}")
    print(f"runs succeeded       : {done}")
    if done:
        print(f"avg makespan (h)     : {mean(mk_):.2f}")
        print(f"avg idle ratio       : {mean(idle_):.2%}")
        print(f"avg completion       : {mean(comp_):.0%}")
        print(f"avg cost ($)         : {mean(cost_):.1f}")
        print(f"avg objective sum    : {mean(obj_):.1f}")
    else:
        print("모든 실험이 스케줄 실패")

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="배치 실험")
    pa.add_argument("--runs", type=int, default=10)
    pa.add_argument("--tasks", type=int, default=5)
    pa.add_argument("--providers", type=int, default=3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--details", type=int, default=1, help="1=세부 출력")
    a, _ = pa.parse_known_args(sys.argv[1:])
    run_batch(a.runs, a.tasks, a.providers, a.seed, bool(a.details))
