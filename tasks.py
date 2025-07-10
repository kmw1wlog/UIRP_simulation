import json
import datetime
from typing import Dict, List, Any, Tuple


class Task:

    def __init__(self, properties: Dict[str, Any]):
        self.properties = properties
        # === Local variables ===
        self.global_file_size: float = properties.get("global_file_size", 0.0)
        self.scene_number: int = properties.get("scene_number", 1)
        self.scene_workload: float = properties.get("scene_workload", 0.0)
        self.bandwidth: float = properties.get("bandwidth", 0.0)
        self.budget: float = properties.get("budget", float("inf"))


        dl = properties.get("deadline")
        self.deadline: datetime.datetime = (
            datetime.datetime.fromisoformat(dl) if isinstance(dl, str) else dl
        )
        start = properties.get("start_time")
        self.start_time: datetime.datetime = (
            datetime.datetime.fromisoformat(start) if isinstance(start, str) else start
        )


class Tasks:
    """Task 컨테이너 (dict)"""
    def __init__(self):
        # Local variable: {task_id: Task}
        self.tasks: Dict[str, Task] = {}
    # -------------------- Public API -------------------------------------
    def initialize_from_data(self, data: List[Dict[str, Any]]) -> None:
        """리스트 형태의 원시 dict 로부터 Task 객체 생성"""
        for item in data:
            tid = item.get("id")
            if tid is None:
                raise ValueError("Task item must have an 'id' field")
            self.tasks[tid] = Task(item)
    # -------------------- Convenience ------------------------------------
    def __iter__(self):
        return iter(self.tasks.values())

    def __getitem__(self, item):
        return self.tasks[item]

if __name__ == "__main__":
    # 샘플 데이터
    sample_data = [
        {
            "id": "task_1",
            "global_file_size": 500.0,
            "scene_number": 3,
            "scene_workload": 150.0,
            "bandwidth": 40.0,
            "budget": 200.0,
            "deadline": "2025-08-01T18:00:00",
            "start_time": "2025-07-25T09:00:00"
        },
        {
            "id": "task_2",
            "global_file_size": 750.0,
            "scene_number": 5,
            "scene_workload": 300.0,
            "bandwidth": 80.0,
            "budget": 400.0,
            "deadline": datetime.datetime(2025, 8, 2, 12, 0, 0),
            "start_time": datetime.datetime(2025, 7, 26, 10, 30, 0)
        }
    ]

    # Tasks 객체 초기화
    tasks = Tasks()
    tasks.initialize_from_data(sample_data)

    # 각 Task의 속성 출력
    for task in tasks:
        print("▶ Task Properties:")
        print(f"  Global File Size: {task.global_file_size} (type: {type(task.global_file_size)})")
        print(f"  Scene Number: {task.scene_number} (type: {type(task.scene_number)})")
        print(f"  Scene Workload: {task.scene_workload} (type: {type(task.scene_workload)})")
        print(f"  Bandwidth: {task.bandwidth} (type: {type(task.bandwidth)})")
        print(f"  Budget: {task.budget} (type: {type(task.budget)})")
        print(f"  Deadline: {task.deadline} (type: {type(task.deadline)})")
        print(f"  Start Time: {task.start_time} (type: {type(task.start_time)})")
        print("-" * 60)



