# ================================================================
# scheduler.py  – 공용 인터페이스 래퍼
# ================================================================
from __future__ import annotations
import datetime
from typing import Protocol, Optional, List, Tuple

from Model.tasks import Tasks
from Model.providers import Providers
from Core.Scheduler.scheduler import BaselineScheduler as _BaselineScheduler

# ---------- 공통 Assignment 형식 ----------
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

# ---------- 스케줄러 최소 프로토콜 ----------
class Scheduler(Protocol):
    def run(
        self,
        tasks: Tasks,
        providers: Providers,
        time_start: Optional[datetime.datetime] = None,
        time_end:   Optional[datetime.datetime] = None,
    ) -> List[Assignment]: ...

# ---------- re-export ----------
BaselineScheduler = _BaselineScheduler

__all__ = ["Scheduler", "Assignment", "BaselineScheduler"]
