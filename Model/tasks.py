# CLEAR

from __future__ import annotations
import datetime
from typing import Dict, List, Tuple, Optional, Any

class Task:
    def __init__(self, d: Dict[str, Any]):
        self.id: str = d["id"]
        self.global_file_size: float = float(d.get("global_file_size", 0.0))  # MB
        self.scene_number: int = int(d.get("scene_number", 1))

        # scene_file_size: 스칼라(모든 씬 동일) 또는 리스트
        sf = d.get("scene_file_size", 0.0)
        self.scene_file_sizes: List[float] = (
            list(map(float, sf)) if isinstance(sf, list)
            else [float(sf)] * self.scene_number
        )
        if len(self.scene_file_sizes) != self.scene_number:
            raise ValueError("scene_file_size 길이와 scene_number 불일치")

        self.scene_workload: float = float(d.get("scene_workload", 0.0))  # GFLOPs 등
        self.bandwidth: float = float(d.get("bandwidth", 0.0))            # MB/s
        self.budget: float = float(d.get("budget", float("inf")))

        self.deadline: datetime.datetime = (
            datetime.datetime.fromisoformat(d["deadline"]) if isinstance(d["deadline"], str)
            else d["deadline"]
        )
        self.start_time: datetime.datetime = (
            datetime.datetime.fromisoformat(d["start_time"]) if isinstance(d["start_time"], str)
            else d["start_time"]
        )

        # (start_time, provider_idx) per scene
        self.scene_allocation_data: List[Tuple[Optional[datetime.datetime], Optional[int]]] = [
            (None, None) for _ in range(self.scene_number)
        ]

    # 헬퍼
    def scene_size(self, idx: int) -> float:
        return self.scene_file_sizes[idx]

class Tasks:
    def __init__(self): self._dict: Dict[str, Task] = {}
    def initialize_from_data(self, data):
        for d in data: self._dict[d["id"]] = Task(d)
    def __iter__(self): return iter(self._dict.values())
    def __getitem__(self, k): return self._dict[k]
