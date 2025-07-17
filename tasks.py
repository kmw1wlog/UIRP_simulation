from __future__ import annotations
import datetime
from typing import Dict, List, Tuple, Optional, Any

class Task:
    def __init__(self, d: Dict[str, Any]):
        self.id = d["id"]
        self.global_file_size = float(d.get("global_file_size", 0.0))
        self.scene_number = int(d.get("scene_number", 1))
        sf = d.get("scene_file_size", 0.0)
        self.scene_file_sizes: List[float] = list(sf) if isinstance(sf, list) else [float(sf)] * self.scene_number
        if len(self.scene_file_sizes) != self.scene_number:
            raise ValueError("scene_file_size length mismatch")
        self.scene_workload = float(d.get("scene_workload", 0.0))
        self.bandwidth = float(d.get("bandwidth", 0.0))
        self.budget = float(d.get("budget", float("inf")))
        self.deadline = datetime.datetime.fromisoformat(d["deadline"]) if isinstance(d["deadline"], str) else d["deadline"]
        self.start_time = datetime.datetime.fromisoformat(d["start_time"]) if isinstance(d["start_time"], str) else d["start_time"]
        self.scene_allocation_data: List[Tuple[Optional[datetime.datetime], Optional[int]]] = [
            (None, None) for _ in range(self.scene_number)
        ]
    def scene_size(self, idx: int) -> float:
        return self.scene_file_sizes[idx]

class Tasks:
    def __init__(self):
        self._dict: Dict[str, Task] = {}
    def initialize_from_data(self, data):
        for d in data:
            self._dict[d["id"]] = Task(d)
    def __iter__(self):
        return iter(self._dict.values())
    def __getitem__(self, k):
        return self._dict[k]
