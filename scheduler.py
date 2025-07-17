
from __future__ import annotations
import datetime
from typing import List, Tuple, Optional

from tasks import Tasks, Task
from providers import Providers
from objective import calc_objective, DEFAULT_WEIGHTS

# (task_id, scene_id, start, finish, provider_idx)
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]


class Scheduler:
    """
    • Earliest-Deadline-First at scene granularity
    • Rendering starts only after the required files have been transferred
    • One GPU → one job enforced by Provider.assign()
    • Candidate provider chosen by minimal calc_objective()
    """

    def __init__(
        self,
        time_gap: datetime.timedelta = datetime.timedelta(minutes=30),
        weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS,
    ):
        self.time_gap = time_gap
        self.weights = weights
        self.waiting: List[Tuple[Task, int]] = []   # (task, scene_id)
        self.results: List[Assignment] = []

    # ----------------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------------
    def _feed(self, now: datetime.datetime, tasks: Tasks) -> None:
        """Push newly-arrived scenes into the EDF queue."""
        seen = {(t.id, s) for t, s in self.waiting}
        for t in tasks:
            if t.start_time <= now:
                for s in range(t.scene_number):
                    if t.scene_allocation_data[s][0] is None and (t.id, s) not in seen:
                        self.waiting.append((t, s))
                        seen.add((t.id, s))
        # EDF order
        self.waiting.sort(key=lambda ts: ts[0].deadline)

    def _schedule_once(
        self, now: datetime.datetime, providers: Providers
    ) -> List[Assignment]:
        """Try to allocate every scene currently in the waiting list."""
        new_assignments: List[Assignment] = []
        remaining: List[Tuple[Task, int]] = []

        for task, scene_id in self.waiting:
            # Skip if allocated concurrently in a previous loop
            if task.scene_allocation_data[scene_id][0] is not None:
                continue

            best: Optional[Tuple[float, int, float, datetime.datetime]] = None
            scene_size = task.scene_size(scene_id)

            for idx, prov in enumerate(providers):
                # --- transfer & compute times ------------------------
                rate = min(task.bandwidth, prov.bandwidth)
                tx_time_h = (task.global_file_size + scene_size) / rate
                cmp_time_h = task.scene_workload / prov.throughput
                earliest_start = prov.earliest_available(
                    cmp_time_h,
                    after=now + datetime.timedelta(hours=tx_time_h),
                )
                if earliest_start is None:
                    continue

                obj = calc_objective(task, scene_id, prov, self.weights)
                if best is None or obj < best[0]:
                    best = (obj, idx, cmp_time_h, earliest_start)

            if best:
                _, idx, dur_h, start = best
                prov = providers[idx]
                prov.assign(task.id, scene_id, start, dur_h)
                finish = start + datetime.timedelta(hours=dur_h)
                task.scene_allocation_data[scene_id] = (start, idx)
                new_assignments.append(
                    (task.id, scene_id, start, finish, idx)
                )
            else:
                remaining.append((task, scene_id))

        self.waiting = remaining
        return new_assignments

    # ----------------------------------------------------------------
    # public entry-point
    # ----------------------------------------------------------------
    def run(
        self,
        tasks: Tasks,
        providers: Providers,
        time_start: Optional[datetime.datetime] = None,
        time_end: Optional[datetime.datetime] = None,
    ) -> List[Assignment]:
        """
        Main simulation loop.

        Parameters
        ----------
        time_start : earliest simulation time (defaults to first task/provider availability)
        time_end   : cutoff (defaults to last task deadline + 1 day)
        """
        if time_start is None:
            earliest_task = min(t.start_time for t in tasks)
            earliest_prov = min(p.available_hours[0][0] for p in providers)
            time_start = min(earliest_task, earliest_prov)

        if time_end is None:
            time_end = max(t.deadline for t in tasks) + datetime.timedelta(days=1)

        now = time_start
        while now < time_end:
            self._feed(now, tasks)
            self.results.extend(self._schedule_once(now, providers))

            # All scenes scheduled → early exit
            if all(
                all(st is not None for st, _ in t.scene_allocation_data) for t in tasks
            ):
                break

            now += self.time_gap

        return self.results

