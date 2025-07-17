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


if __name__ == "__main__":
    import datetime as dt

    # ---------- Test 1: scene_file_size provided as **list** ----------
    data1 = {
        "id": "task1",
        "global_file_size": 123.4,
        "scene_number": 3,
        "scene_file_size": [10, 20, 30],
        "scene_workload": 55.5,
        "bandwidth": 1.25,
        "budget": 999,
        "deadline": "2025-07-31T12:00:00",
        "start_time": "2025-07-30T00:00:00",
    }
    tasks1 = Tasks()
    tasks1.initialize_from_data([data1])
    t1 = tasks1["task1"]

    assert t1.scene_file_sizes == [10.0, 20.0, 30.0]
    assert t1.scene_size(2) == 30.0
    assert len(t1.scene_allocation_data) == 3
    assert t1.deadline == dt.datetime(2025, 7, 31, 12, 0, 0)
    assert t1.start_time == dt.datetime(2025, 7, 30, 0, 0, 0)

    # ---------- Test 2: scene_file_size provided as **scalar** ----------
    data2 = {
        "id": "task2",
        "scene_number": 2,
        "scene_file_size": 5,          # scalar → duplicated
        "deadline": dt.datetime(2025, 8, 1, 0, 0, 0),
        "start_time": dt.datetime(2025, 7, 31, 0, 0, 0),
    }
    tasks2 = Tasks()
    tasks2.initialize_from_data([data2])
    t2 = tasks2["task2"]

    assert t2.scene_file_sizes == [5.0, 5.0]
    assert t2.scene_size(1) == 5.0
    assert len(t2.scene_allocation_data) == 2

    # ---------- Test 3: length mismatch triggers ValueError ----------
    try:
        Task({
            "id": "bad",
            "scene_number": 2,
            "scene_file_size": [1],    # wrong length
            "deadline": "2025-01-01T00:00:00",
            "start_time": "2025-01-01T00:00:00",
        })
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised for length mismatch")

    print("all Task/Tasks tests passed")