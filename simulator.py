# simulator.py
# CLEAR

from __future__ import annotations
import json, datetime, pprint, argparse
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from Model.tasks import Tasks, Task
from Model.providers import Providers
from Core.scheduler import BaselineScheduler, Assignment
from utils.utils import merge_intervals

_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def _task_color_map(task_ids):
    return {tid: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, tid in enumerate(task_ids)}

def _is_light(rgb_hex):
    r, g, b = mcolors.to_rgb(rgb_hex)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.5


class Simulator:
    def __init__(self, cfg_path: str):
        cfg = json.loads(Path(cfg_path).read_text())
        self.tasks = Tasks();      self.tasks.initialize_from_data(cfg["tasks"])
        self.providers = Providers(); self.providers.initialize_from_data(cfg["providers"])
        self.results: List[Assignment] = []
        self.evaluator = None  # BaselineScheduler.evaluator 주입용

    # scheduler class 에 tasks, providers 전달, run 메서드 실행
    def schedule(self, sch: BaselineScheduler):
        # evaluator 가중치를 evaluate에서 쓰기 위해 보관
        self.evaluator = getattr(sch, "evaluator", None)
        self.results = sch.run(self.tasks, self.providers)

    # Task 에 대하여, 스케줄링 결과 완료되지 않은 씬 정보
    @staticmethod
    def _missing(task: Task) -> List[int]:
        return [i for i, (st, _) in enumerate(task.scene_allocation_data) if st is None]

    def _idle(self) -> float:
        """
        전체 '유틸(=busy) 비율'을 다음으로 계산:
          sum_scheduled / (sum_remaining_available + sum_scheduled)
        주의: Provider.assign()이 available_hours를 즉시 깎기 때문에,
             남은 available_hours + 배정된 schedule 시간이 원래 총 가용량.
        """
        total_sched_h = 0.0
        total_remain_avail_h = 0.0
        for p in self.providers:
            # 배정된 총 시간
            busy = merge_intervals([(st, ft) for _, _, st, ft in p.schedule])
            sched_h = sum((b2 - b1).total_seconds() / 3600.0 for b1, b2 in busy)
            total_sched_h += sched_h
            # 남은 가용 시간 (assign 이후 남은 available_hours 전체)
            total_remain_avail_h += sum(
                (e - s).total_seconds() / 3600.0 for s, e in getattr(p, "available_hours", [])
            )
        denom = total_remain_avail_h + total_sched_h
        return (total_sched_h / denom) if denom > 0 else 0.0

    def evaluate(self) -> Dict[str, Any]:
        if not self.results:
            raise RuntimeError("schedule() 먼저 호출 필요")

        # evaluator 가중치 (없으면 기본값)
        WT  = getattr(self.evaluator, "WT", 1.0)
        WC  = getattr(self.evaluator, "WC", 1.0)
        WD  = getattr(self.evaluator, "WD", 10.0)
        WB  = getattr(self.evaluator, "WB", 200.0)
        WDL = getattr(self.evaluator, "WDL", 500.0)

        tasks_out: Dict[str, Any] = {}
        finished: List[Assignment] = []
        tot_cost = 0.0
        tot_obj  = 0.0

        # 태스크별 결과 집계
        for t in self.tasks:
            rec = [r for r in self.results if r[0] == t.id]
            miss = self._missing(t)

            # 비용
            cost = sum((r[3] - r[2]).total_seconds() / 3600.0 *
                       self.providers[r[4]].price_per_gpu_hour for r in rec)
            tot_cost += cost

            # 완료 여부 및 완료시간
            if rec:
                starts   = [r[2] for r in rec]
                finishes = [r[3] for r in rec]
                completion_h = (max(finishes) - min(starts)).total_seconds() / 3600.0
                lateness_h   = max(0.0, (max(finishes) - t.deadline).total_seconds() / 3600.0)
            else:
                completion_h = 0.0
                lateness_h   = 0.0

            budget_over = max(0.0, cost - t.budget)
            deferred_cnt = len(miss)

            # 소프트 패널티를 포함한 점수 합산
            tot_obj += (WT * completion_h +
                        WC * cost +
                        WB * budget_over +
                        WDL * lateness_h +
                        WD * deferred_cnt)

            if miss:
                tasks_out[t.id] = {
                    "completed": False,
                    "missing": miss,
                    "cost": cost,
                    "budget_ok": cost <= t.budget,
                    "budget_over": budget_over,
                    "lateness_h": lateness_h,
                }
            else:
                tasks_out[t.id] = {
                    "completed": True,
                    "completion_h": completion_h,
                    "cost": cost,
                    "budget_ok": cost <= t.budget,
                    "budget_over": budget_over,
                    "deadline_ok": lateness_h == 0.0,
                    "lateness_h": lateness_h,
                }
                finished += rec

        # 전체 메이크스팬(완료된 작업 기준)
        makespan = ((max(r[3] for r in finished) - min(r[2] for r in finished)).total_seconds() / 3600.0
                    if finished else 0.0)

        return {
            "tasks": tasks_out,
            "makespan_h": makespan,
            "overall_idle_ratio": self._idle(),  # 위 설명대로 정의된 유틸 비율
            "total_cost": tot_cost,
            "objective_sum": tot_obj,            # WT/WC/WB/WDL/WD 기반 합산 점수
        }

    def visualize(self, save_path: str | None = None, show: bool = True,
                  figsize: tuple[int, int] | None = None):
        if not self.results:
            raise RuntimeError("schedule() 먼저 호출 필요")

        if figsize is None:
            figsize = (14, max(3, 1 + 0.8 * len(self.providers)))

        # --- 1. 모든 provider의 availability 를 포함해 절대 최소/최대 시각 구하기
        # 주의: available_hours는 assign() 이후 '남은' 가용시간 구간들임
        avail_points = [ts for p in self.providers for ts in p.available_hours]
        if avail_points:
            avail_min = min(s for (s, _) in avail_points)
            avail_max = max(e for (_, e) in avail_points)
        else:
            # 가용구간이 모두 소진되었을 수도 있으니, 결과 기준으로 fallback
            avail_min = min(r[2] for r in self.results)
            avail_max = max(r[3] for r in self.results)

        # --- 2. results 가 없을 때를 대비한 세이프가드(이미 위에서 체크함)
        task_min = min(r[2] for r in self.results)
        task_max = max(r[3] for r in self.results)

        start_min = min(avail_min, task_min)
        finish_max = max(avail_max, task_max)

        task_ids = [t.id for t in self.tasks]
        color_map = _task_color_map(task_ids)

        prices = [prov.price_per_gpu_hour for prov in self.providers]
        max_price = max(prices) if prices else 1.0

        fig, ax = plt.subplots(figsize=figsize)

        # 남은 가용창(회색)을 먼저 그림
        for prov_idx, prov in enumerate(self.providers):
            shade_norm = prov.price_per_gpu_hour / max_price
            shade = 1 - shade_norm
            for avail_start, avail_end in prov.available_hours:
                st_num = mdates.date2num(avail_start)
                ft_num = mdates.date2num(avail_end)
                ax.add_patch(Rectangle(
                    (st_num, prov_idx - 0.5),
                    ft_num - st_num,
                    1.0,
                    facecolor=(shade, shade, shade),
                    alpha=0.3,
                    zorder=0,
                    linewidth=0
                ))
                ax.plot(st_num, prov_idx, marker="^", color="black", markersize=6, zorder=5,
                        label="Availability Start" if prov_idx == 0 else "")
                ax.plot(ft_num, prov_idx, marker="v", color="black", markersize=6, zorder=5,
                        label="Availability End" if prov_idx == 0 else "")

        # 실제 배정된 블록을 겹쳐서 그림
        for prov_idx, _ in enumerate(self.providers):
            prov_results = [r for r in self.results if r[4] == prov_idx]
            for task_id, scene_idx, st, ft, _ in prov_results:
                st_num, ft_num = mdates.date2num(st), mdates.date2num(ft)
                facecolor = color_map[task_id]
                ax.add_patch(Rectangle(
                    (st_num, prov_idx - 0.4),
                    ft_num - st_num, 0.8,
                    facecolor=facecolor, edgecolor="black", linewidth=0.5,
                    alpha=0.8, zorder=2, label=task_id
                ))
                text_color = "black" if _is_light(facecolor) else "white"
                ax.text((st_num + ft_num) / 2, prov_idx,
                        f"{task_id}\nS{scene_idx}",
                        va="center", ha="center", fontsize=7,
                        color=text_color, zorder=3)

        total_span = (finish_max - start_min).total_seconds() / 3600.0
        pad_h = total_span * 0.05
        ax.set_xlim(mdates.date2num(start_min - datetime.timedelta(hours=pad_h)),
                    mdates.date2num(finish_max + datetime.timedelta(hours=pad_h)))
        ax.set_ylim(-0.7, len(self.providers) - 1 + 0.7)

        ax.set_yticks(range(len(self.providers)))
        ax.set_yticklabels([f"Prov {i}" for i in range(len(self.providers))])
        ax.set_xlabel("Time"); ax.set_ylabel("Provider")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.autofmt_xdate()
        handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Task", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title("Schedule Visualization"); ax.grid(True, axis="x", linestyle=":", linewidth=0.5, zorder=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✔ Schedule image saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)

# ---------------- CLI ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="config.json")
    pa.add_argument("--algo",   default="bf", help="bf | cp")
    pa.add_argument("--out-img", default="schedule.png", help="Output image path")
    pa.add_argument("-v", action="count", default=0, help="-v / -vv for verbose")
    args = pa.parse_args()

    sim = Simulator(args.config)
    sch = BaselineScheduler(algo=args.algo,
                            verbose=args.v >= 1,
                            time_gap=datetime.timedelta(hours=1))
    sim.schedule(sch)
    pprint.pprint(sim.evaluate(), sort_dicts=False)
    sim.visualize(save_path=args.out_img, show=True)
