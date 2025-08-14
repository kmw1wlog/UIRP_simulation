from __future__ import annotations
import datetime
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence

from Model.tasks import Tasks, Task
from Model.providers import Providers, Provider

Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

class TaskSelector(ABC):
    @abstractmethod
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]: ...

class ComboGenerator(ABC):
    @abstractmethod
    def best_combo(
        self, task: Task, providers: Providers, sim_time: datetime.datetime,
        evaluator: "MetricEvaluator", verbose: bool = False,
    ) -> Optional[Tuple[List[int], float, float]]: ...

class MetricEvaluator(ABC):
    @abstractmethod
    def time_cost(self, task: Task, scene_id: int, prov: Provider) -> Tuple[float, float]: ...

    @abstractmethod
    def feasible(self, task: Task, combo: List[int], sim_time: datetime.datetime,
                 providers: Providers) -> Tuple[bool, float, float]: ...

    @abstractmethod
    def efficiency(self, task: Task, combo: List[int], ps: Providers, now: datetime.datetime) -> float:
        ...

class Dispatcher(ABC):
    @abstractmethod
    def dispatch(
        self, task: Task, combo: List[int], sim_time: datetime.datetime,
        providers: Providers, evaluator: MetricEvaluator, verbose: bool
    ) -> List[Assignment]: ...