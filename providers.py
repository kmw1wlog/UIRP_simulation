from __future__ import annotations
import datetime
from typing import Dict, Any, List, Tuple, Optional
from utils import merge_intervals

class Provider:
    def __init__(self, d: Dict[str, Any]):
        self.throughput = float(d.get("throughput", 1.0))
        self.price_per_gpu_hour = float(d.get("price", 0.0))
        self.bandwidth = float(d.get("bandwidth", 0.0))
        raw = d.get("available_hours", [])
        self.available_hours: List[Tuple[datetime.datetime, datetime.datetime]] = [
            (datetime.datetime.fromisoformat(s) if isinstance(s, str) else s,
             datetime.datetime.fromisoformat(e) if isinstance(e, str) else e)
            for s, e in raw
        ]
        self.schedule: List[Tuple[str, int, datetime.datetime, datetime.datetime]] = []

    def idle_ratio(self, start: Optional[datetime.datetime] = None,
                   end: Optional[datetime.datetime] = None) -> float:
        if not self.schedule:
            return 1.0
        if start is None:
            start = min(s for _, _, s, _ in self.schedule)
        if end is None:
            end = max(f for _, _, _, f in self.schedule)
        horizon = (end - start).total_seconds() / 3600.0
        if horizon <= 0:
            return 1.0
        busy = merge_intervals([(s, f) for _, _, s, f in self.schedule])
        busy_h = sum((f - s).total_seconds() / 3600.0 for s, f in busy)
        return max(0.0, 1.0 - busy_h / horizon)

    def earliest_available(self, dur_h: float, after: datetime.datetime) -> Optional[datetime.datetime]:
        for a_s, a_e in sorted(self.available_hours, key=lambda t: t[0]):
            if a_e <= after:
                continue
            cur = max(a_s, after)
            clashes = sorted([(s, f) for _, _, s, f in self.schedule if s < a_e and f > cur], key=lambda iv: iv[0])
            for s, f in clashes:
                if (s - cur).total_seconds() / 3600.0 >= dur_h:
                    return cur
                cur = max(cur, f)
            if (a_e - cur).total_seconds() / 3600.0 >= dur_h:
                return cur
        return None

    def assign(self, task_id: str, scene_id: int, start: datetime.datetime, dur_h: float):
        finish = start + datetime.timedelta(hours=dur_h)
        self.schedule.append((task_id, scene_id, start, finish))
        new = []
        for s, e in self.available_hours:
            if finish <= s or start >= e:
                new.append((s, e))
            else:
                if s < start:
                    new.append((s, start))
                if finish < e:
                    new.append((finish, e))
        self.available_hours = new

class Providers:
    def __init__(self):
        self._list: List[Provider] = []
    def initialize_from_data(self, data):
        for d in data:
            self._list.append(Provider(d))
    def __iter__(self):
        return iter(self._list)
    def __getitem__(self, i):
        return self._list[i]
    def __len__(self):
        return len(self._list)


if __name__ == "__main__":
    import datetime as dt
    import math

    # ----------------------------------------------------------
    # Helper to build ISO-8601 strings quickly
    iso = lambda y, m, d, H=0: f"{y:04d}-{m:02d}-{d:02d}T{H:02d}:00:00"
    # ----------------------------------------------------------

    # ---------- Test 1: basic construction & idle_ratio (empty) ----------
    p_data = [{
        "throughput": 1.5,
        "price": 0.12,
        "bandwidth": 10,
        "available_hours": [
            (iso(2025, 1, 1, 9), iso(2025, 1, 1, 14)),   # 09-14
        ],
    }]
    providers = Providers()
    providers.initialize_from_data(p_data)
    p = providers[0]

    assert len(providers) == 1
    assert list(iter(providers))[0] is p
    assert math.isclose(p.idle_ratio(), 1.0), "idle_ratio should be 1.0 when no schedule"

    # ---------- Test 2: earliest_available before any assignment ----------
    start1 = p.earliest_available(dur_h=3, after=dt.datetime(2025, 1, 1, 8))
    assert start1 == dt.datetime(2025, 1, 1, 9), "first 3-hour slot should start at 09:00"

    # ---------- Test 3: assign first job & availability update ----------
    p.assign("taskA", 0, start1, dur_h=3)  # 09-12
    assert p.available_hours == [
        (dt.datetime(2025, 1, 1, 12), dt.datetime(2025, 1, 1, 14)),
    ], "available_hours should be split after assignment (09-12 booked)"

    # idle_ratio over the occupied horizon (09-12) should be 0
    assert math.isclose(p.idle_ratio(), 0.0), "fully busy during 09-12 horizon => idle 0.0"

    # ---------- Test 4: next earliest slot & second assignment ----------
    start2 = p.earliest_available(dur_h=1, after=dt.datetime(2025, 1, 1, 11))
    assert start2 == dt.datetime(2025, 1, 1, 12), "next 1-hour slot should begin at 12:00"

    p.assign("taskB", 1, start2, dur_h=1)  # 12-13
    assert p.available_hours == [
        (dt.datetime(2025, 1, 1, 13), dt.datetime(2025, 1, 1, 14)),
    ], "after 12-13 booking, remaining availability should be 13-14"

    # ---------- Test 5: idle_ratio over wider horizon ----------
    ratio = p.idle_ratio(
        start=dt.datetime(2025, 1, 1, 9),
        end=dt.datetime(2025, 1, 1, 14),
    )
    # Busy = 09-13 (4 h) out of 09-14 (5 h) → idle = 1/5 = 0.2
    assert math.isclose(ratio, 0.2), f"idle_ratio over 5-hour horizon should be 0.2, got {ratio}"

    # ---------- Test 6: earliest_available returns None when no room ----------
    start3 = p.earliest_available(dur_h=2, after=dt.datetime(2025, 1, 1, 13))
    assert start3 is None, "no 2-hour block fits into remaining 13-14 window"

    print("all Provider/Providers tests passed")
