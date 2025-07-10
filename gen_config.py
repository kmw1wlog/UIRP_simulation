# gen_config.py
import json
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any

# ---------- 하드코딩된 기준일 ----------
BASE_DAY_1 = datetime.datetime(2025, 7, 12)  # 첫날 00:00
BASE_DAY_2 = BASE_DAY_1 + datetime.timedelta(days=1)

# ---------- 가우스 샘플 + 클램핑 ----------
def g(mean: float, std: float, lo: float, hi: float) -> float:
    """정규분포 샘플 → [lo, hi] 로 제한"""
    return max(lo, min(hi, random.gauss(mean, std)))

# ---------- 태스크 & 프로바이더 생성 함수 ----------
def make_tasks(n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(1, n + 1):
        st = BASE_DAY_1 + datetime.timedelta(
            hours=g(8, 2, 0, 18),            # 0~18시 사이
            minutes=g(0, 30, 0, 59)
        )
        dl = st + datetime.timedelta(
            hours=g(36, 12, 12, 96)          # 최소 12h, 최대 4일
        )

        t = {
            "id": f"task_{i}",
            "global_file_size": round(g(600, 150, 100, 1200), 1),   # MB
            "scene_number": int(g(4, 1.2, 1, 8)),
            "scene_workload": round(g(180, 60, 30, 400), 1),        # GPU·h
            "bandwidth": round(g(60, 15, 10, 200), 1),              # MB/s
            "budget": round(g(450, 120, 100, 2000), 1),             # $
            "deadline": dl.isoformat(timespec="seconds"),
            "start_time": st.isoformat(timespec="seconds"),
        }
        out.append(t)
    return out


def make_providers(n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(1, n + 1):
        intervals = []
        for base_day in (BASE_DAY_1, BASE_DAY_2):
            start = base_day + datetime.timedelta(
                hours=g(8 + i, 1.5, 6, 20))          # 겹치지 않게 약간 시프트
            dur_h = g(6, 2, 3, 12)
            end = start + datetime.timedelta(hours=dur_h)
            intervals.append([
                start.isoformat(timespec="seconds"),
                end.isoformat(timespec="seconds")
            ])

        p = {
            "throughput": round(g(30, 10, 5, 100), 1),   # 작업 속도
            "available_hours": intervals,
            "price": round(g(5, 1,   1, 15), 2),         # $/GPU·h
            "bandwidth": round(g(90, 20, 20, 400), 1),   # MB/s
        }
        out.append(p)
    return out

# ---------- 메인 ----------
def generate_cfg(
    n_tasks: int = 5,
    n_providers: int = 3,
    seed: int = 42,
    filepath: str = "config_generated.json"
) -> None:
    random.seed(seed)
    cfg = {
        "tasks": make_tasks(n_tasks),
        "providers": make_providers(n_providers),
    }
    Path(filepath).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✔ 생성 완료 → {filepath}")

if __name__ == "__main__":
    # Jupyter 등 불필요한 인자를 무시하도록 parse_known_args 사용
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="EDF 시뮬레이션용 config.json 생성기",
        add_help=True
    )
    parser.add_argument("--tasks", type=int, default=5,
                        help="생성할 Task 개수")
    parser.add_argument("--providers", type=int, default=3,
                        help="생성할 Provider 개수")
    parser.add_argument("--seed", type=int, default=42,
                        help="난수 시드")
    parser.add_argument("--out", type=str, default="config_generated.json",
                        help="출력 파일명")
    args, _ = parser.parse_known_args()

    generate_cfg(
        n_tasks=args.tasks,
        n_providers=args.providers,
        seed=args.seed,
        filepath=args.out
    )
