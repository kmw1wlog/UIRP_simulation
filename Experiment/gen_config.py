# gen_config.py
import json, random, datetime, argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

DEFAULT_BASE_DAY = datetime.datetime(2024, 1, 1, 6, 0, 0)

def _g(mu, sig, lo, hi) -> float:
    import random
    x = random.gauss(mu, sig)
    return max(lo, min(hi, x))

def _rand_time(base: datetime.datetime, *, lo_h: float, hi_h: float) -> datetime.datetime:
    return base + datetime.timedelta(hours=random.uniform(lo_h, hi_h))

def _interval(base: datetime.datetime, *, start_lo=0, start_hi=6, dur_lo=10, dur_hi=16):
    s = _rand_time(base, lo_h=start_lo, hi_h=start_hi)
    e = s + datetime.timedelta(hours=random.uniform(dur_lo, dur_hi))
    return (s.isoformat(timespec="seconds"), e.isoformat(timespec="seconds"))

def _mk_task(idx: int, base_day: datetime.datetime) -> Dict[str, Any]:
    scenes = int(round(_g(4.0, 1.5, 1, 8)))
    global_mb = round(_g(600.0, 200.0, 100.0, 2000.0), 1)
    scene_base = _g(200.0, 60.0, 20.0, 600.0)
    scene_files = [round(_g(scene_base, 25.0, 10.0, 800.0), 1) for _ in range(scenes)]
    bandwidth = round(_g(150.0, 60.0, 30.0, 600.0), 1)  # MB/s
    scene_workload = round(_g(2.5, 1.2, 0.5, 6.0), 2)
    start = _rand_time(base_day, lo_h=1, hi_h=18)
    deadline = start + datetime.timedelta(hours=_g(24.0, 8.0, 6.0, 48.0))
    avg_price = 30.0
    budget_est = scenes * scene_workload * avg_price * _g(1.6, 0.2, 1.2, 2.2)
    budget = round(max(50.0, budget_est), 2)

    return {
        "id": f"task_{idx:03d}",
        "global_file_size": global_mb,
        "scene_number": scenes,
        "scene_file_size": scene_files,
        "scene_workload": scene_workload,
        "bandwidth": bandwidth,
        "budget": budget,
        "deadline": deadline.isoformat(timespec="seconds"),
        "start_time": start.isoformat(timespec="seconds"),
    }

def _mk_provider(idx: int, base_day: datetime.datetime) -> Dict[str, Any]:
    throughput = int(round(_g(8.0, 4.0, 2.0, 32.0)))
    price = round(_g(28.0, 7.0, 10.0, 80.0), 2)
    bandwidth = round(_g(200.0, 80.0, 50.0, 1200.0), 1)
    iv1 = _interval(base_day, start_lo=0, start_hi=4, dur_lo=10, dur_hi=16)
    iv2 = _interval(base_day + datetime.timedelta(days=1), start_lo=0, start_hi=6, dur_lo=12, dur_hi=18)
    return {
        "throughput": throughput,
        "price": price,
        "bandwidth": bandwidth,
        "available_hours": [iv1, iv2],
    }

def generate_cfg(n_tasks: int, n_prov: int, seed: int, out_path: str = "config.json",
                 base_day: datetime.datetime = DEFAULT_BASE_DAY) -> None:
    random.seed(seed)
    tasks = [_mk_task(i+1, base_day) for i in range(n_tasks)]
    providers = [_mk_provider(i+1, base_day) for i in range(n_prov)]
    cfg = {"tasks": tasks, "providers": providers}
    Path(out_path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

def _parse_args(argv: List[str]):
    ap = argparse.ArgumentParser(description="랜덤 config.json 생성기 (개선판, 코드베이스 호환)")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--providers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="config.json")
    ap.add_argument("--base-day", type=str, default=None)
    # parse_known_args 로 -f 등 무시
    args, _unknown = ap.parse_known_args(argv)
    return args

if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    base_day = (datetime.datetime.fromisoformat(args.base_day)
                if args.base_day else DEFAULT_BASE_DAY)
    generate_cfg(args.tasks, args.providers, args.seed, args.out, base_day)
    print(f"✔ {args.out} 생성 완료")
