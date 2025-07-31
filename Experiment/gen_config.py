# gen_config.py
import json, random, datetime, argparse, sys
from pathlib import Path
from typing import List, Dict, Any

BASE_DAY = datetime.datetime(2025, 7, 12)

def _g(mu, sig, lo, hi):                 # 가우스 샘플 + 클램프
    import math, random
    return max(lo, min(hi, random.gauss(mu, sig)))

def _interval(base):
    s = base + datetime.timedelta(hours=_g(8, 2, 6, 21))
    e = s   + datetime.timedelta(hours=_g(6, 2, 3, 12))
    return [s.isoformat(timespec="seconds"), e.isoformat(timespec="seconds")]

def _make_tasks(n):
    tasks = []
    for i in range(1, n+1):
        st = BASE_DAY + datetime.timedelta(hours=_g(6, 4, 0, 30))
        dl = st + datetime.timedelta(hours=_g(60, 20, 24, 120))
        scenes = int(_g(3, 1, 1, 6))
        base = _g(140, 40, 20, 400)
        tasks.append({
            "id": f"task_{i}",
            "global_file_size": round(_g(550,150,100,1400),1),
            "scene_number": scenes,
            "scene_file_size": [round(_g(base,12,10,600),1) for _ in range(scenes)],
            "scene_workload": round(_g(60,20,15,120),1),
            "bandwidth": round(_g(90,25,20,400),1),
            "budget": round(_g(800,250,200,5000),1),
            "deadline": dl.isoformat(timespec="seconds"),
            "start_time": st.isoformat(timespec="seconds"),
        })
    return tasks

def _make_providers(n):
    provs=[]
    for _ in range(n):
        provs.append({
            "throughput": round(_g(40,10,8,120),1),
            "available_hours":[
                _interval(BASE_DAY),
                _interval(BASE_DAY+datetime.timedelta(days=1))
            ],
            "price": round(_g(5,1,1,15),2),
            "bandwidth": round(_g(110,30,30,800),1),
        })
    return provs

def generate_cfg(n_tasks:int, n_prov:int, seed:int, out_path:str="config.json"):
    random.seed(seed)
    cfg = {"tasks": _make_tasks(n_tasks), "providers": _make_providers(n_prov)}
    Path(out_path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="랜덤 config.json 생성기")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--providers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,  default="config.json")
    args, _ = ap.parse_known_args(sys.argv[1:])   # Jupyter -f 무시
    generate_cfg(args.tasks, args.providers, args.seed, args.out)
    print(f"✔ {args.out} 생성 완료")
