# run_experiments.py
"""
여러 차례 무작위 config.json을 생성 → Simulator로 실행 → 성능 지표 집계.

사용 예
------
$ python run_experiments.py --runs 20 --tasks 6 --providers 4 --seed 2025 --details 0
"""
import importlib
import json
from pathlib import Path
from statistics import mean
import argparse
import tempfile

# --------------- 동적 import ----------------
gen_config = importlib.import_module("gen_config")      # config 생성기
sim_mod    = importlib.import_module("simulator")      # Simulator 포함
Simulator  = sim_mod.Simulator
Scheduler  = sim_mod.Scheduler

# --------------- 실험 루프 ------------------
def run_batch(
    runs: int = 10,
    n_tasks: int = 5,
    n_providers: int = 3,
    seed: int = 42,
    show_details: bool = True,
) -> None:
    """runs 번 반복 실행 후 메트릭 요약"""
    all_makespan, all_idle, completed_rates = [], [], []

    for i in range(runs):
        # -------- config 생성 --------
        cfg_path = Path(tempfile.gettempdir()) / f"cfg_{i}.json"
        gen_config.generate_cfg(
            n_tasks=n_tasks,
            n_providers=n_providers,
            seed=seed + i,
            filepath=str(cfg_path),
        )

        # -------- 시뮬레이션 --------
        sim = Simulator(str(cfg_path))
        sim.schedule(Scheduler())
        metrics = sim.evaluate()

        # -------- 개별 결과 출력 --------
        makespan  = metrics["makespan_h"]
        idle      = metrics["overall_idle_ratio"]
        task_stats = metrics["tasks"]
        completed = sum(1 for v in task_stats.values() if v["completed"])
        comp_rate = completed / len(task_stats)

        header = f"[#{i:02}] makespan {makespan:.1f}h | idle {idle:.2%} | " \
                 f"completed {completed}/{len(task_stats)} ({comp_rate:.0%})"
        print(header)

        if show_details:
            print(f"  ├─ config : {cfg_path}")
            print("  ├─ schedule (task_id, scene, start, finish, provider_idx):")
            print(json.dumps(sim.results, default=str, indent=2, ensure_ascii=False))
            print("  ╰─ metrics:")
            print(json.dumps(metrics, default=str, indent=2, ensure_ascii=False))
            print("-" * 80)

        all_makespan.append(makespan)
        all_idle.append(idle)
        completed_rates.append(comp_rate)

    # -------- 전체 요약 --------
    if all_makespan:
        print("\n=== Overall Summary ===")
        print(f"runs                : {runs}")
        print(f"avg makespan        : {mean(all_makespan):.1f} h")
        print(f"avg idle ratio      : {mean(all_idle):.2%}")
        print(f"avg completion rate : {mean(completed_rates):.0%}")

# --------------- CLI ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator 배치 실험")
    parser.add_argument("--runs", type=int, default=10, help="실험 반복 횟수")
    parser.add_argument("--tasks", type=int, default=5, help="태스크 개수")
    parser.add_argument("--providers", type=int, default=3, help="프로바이더 개수")
    parser.add_argument("--seed", type=int, default=42, help="초기 난수 시드")
    parser.add_argument("--details", type=int, default=1,
                        help="1: 각 실험 세부 결과 전체 출력, 0: 요약만 출력")
    args, _ = parser.parse_known_args()

    run_batch(
        runs=args.runs,
        n_tasks=args.tasks,
        n_providers=args.providers,
        seed=args.seed,
        show_details=bool(args.details),
    )
