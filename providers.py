import datetime
from typing import Dict, List, Any, Tuple, Optional


class Provider:
    """GPU 제공자 노드"""
    def __init__(self, properties: Dict[str, Any]):
        self.properties = properties

        # === Local variables ===
        self.throughput: float = float(properties.get("throughput", 1.0))
        self.price_per_gpu_hour: float = float(properties.get("price", 0.0))
        self.bandwidth: float = float(properties.get("bandwidth", 0.0))

        # Available hours: 문자열 → datetime 변환
        raw_hours = properties.get("available_hours", [])
        self.available_hours: List[Tuple[datetime.datetime, datetime.datetime]] = []
        for s, e in raw_hours:
            try:
                start_dt = datetime.datetime.fromisoformat(s) if isinstance(s, str) else s
                end_dt = datetime.datetime.fromisoformat(e) if isinstance(e, str) else e
                if start_dt >= end_dt:
                    raise ValueError("Start time must be before end time.")
                self.available_hours.append((start_dt, end_dt))
            except Exception as ex:
                raise ValueError(f"Invalid available_hours format: {s}, {e}") from ex

        # 스케줄: (task_id, scene_id, start, finish)
        self.schedule: List[Tuple[str, int, datetime.datetime, datetime.datetime]] = []

    def earliest_available(self, duration_h: float) -> Optional[datetime.datetime]:
        """예약된 시간 포함하여, 가장 빠른 사용 가능한 시작 시간 반환"""
        now = datetime.datetime.now()

        for start, end in sorted(self.available_hours, key=lambda t: t[0]):
            if end <= now:
                continue

            check_start = max(now, start)
            check_end = end
            # scene_id 포함으로 수정
            busy_slots = sorted([(s, f) for _, _, s, f in self.schedule if s < check_end and f > check_start])

            candidate_start = check_start
            for res_start, res_end in busy_slots:
                gap_h = (res_start - candidate_start).total_seconds() / 3600
                if gap_h >= duration_h:
                    return candidate_start
                candidate_start = max(candidate_start, res_end)

            if (check_end - candidate_start).total_seconds() / 3600 >= duration_h:
                return candidate_start

        return None

    def assign(self, task_id: str, scene_id: int, start: datetime.datetime, duration_h: float) -> None:
        """예약 추가 및 가용 시간 차감"""
        finish = start + datetime.timedelta(hours=duration_h)
        self.schedule.append((task_id, scene_id, start, finish))

        # available_hours 업데이트
        new_hours = []
        for avail_start, avail_end in self.available_hours:
            if finish <= avail_start or start >= avail_end:
                new_hours.append((avail_start, avail_end))
            else:
                if avail_start < start:
                    new_hours.append((avail_start, start))
                if finish < avail_end:
                    new_hours.append((finish, avail_end))
        self.available_hours = new_hours


class Providers:
    """Provider 컨테이너"""
    def __init__(self):
        self.providers: List[Provider] = []

    def initialize_from_data(self, data: List[Dict[str, Any]]) -> None:
        for item in data:
            self.providers.append(Provider(item))

    def __iter__(self):
        return iter(self.providers)

    def __getitem__(self, index: int):
        return self.providers[index]


# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    import pprint

    sample_data = [
        {
            "throughput": 2.5,
            "available_hours": [
                ("2025-07-12T08:00:00", "2025-07-12T12:00:00"),
                ("2025-07-13T14:00:00", "2025-07-13T18:00:00"),
            ],
            "price": 4.0,
            "bandwidth": 100.0
        },
        {
            "throughput": 3.0,
            "available_hours": [
                ("2025-07-12T10:00:00", "2025-07-12T15:00:00"),
            ],
            "price": 6.0,
            "bandwidth": 80.0
        }
    ]

    providers = Providers()
    providers.initialize_from_data(sample_data)

    print("\n▶ Providers 상태:")
    for i, p in enumerate(providers):
        print(f"Provider {i + 1}")
        pprint.pprint(p.available_hours)

    # 첫 씬 예약 시도
    duration = 2.0
    earliest = providers[0].earliest_available(duration)
    print(f"\nProvider 1 earliest for {duration}h: {earliest}")

    # 예약 실행: task_id="task_1", scene_id=0
    if earliest:
        providers[0].assign("task_1", scene_id=0, start=earliest, duration_h=duration)
        print(f"\nAfter assigning 'task_1', scene 0 from {earliest}:")
        pprint.pprint(providers[0].available_hours)
        print(f"Schedule:")
        pprint.pprint(providers[0].schedule)

    # 두 번째 씬 예약 시도
    earliest2 = providers[0].earliest_available(duration)
    print(f"\nProvider 1 next earliest for {duration}h: {earliest2}")
