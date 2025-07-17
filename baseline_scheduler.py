from __future__ import annotations
import datetime
from typing import List, Tuple, Optional, Dict
from itertools import product

from tasks import Tasks, Task
from providers import Providers, Provider

# (task_id, scene_id, start, finish, provider_idx)
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]


class BaselineScheduler:
    """
    베이스라인 스케줄러 - 사용자 선택 모방
    • FIFO Task 처리 (start_time 순)
    • GPU 중복 허용 (같은 GPU에 여러 씬 순차 배정)
    • 효율성 최대화: (scene_number × scene_workload) / (C_total × T_tot)
    • 예산/데드라인 필터링
    """

    def __init__(
            self,
            time_gap: datetime.timedelta = datetime.timedelta(hours=1),
            max_combinations: int = 50,  # 조합 수 제한
            verbose: bool = False,  # 디버그 출력 제어
    ):
        self.time_gap = time_gap
        self.max_combinations = max_combinations
        self.verbose = verbose
        self.waiting_tasks: List[Task] = []  # Task 단위로 대기
        self.results: List[Assignment] = []

    # ----------------------------------------------------------------
    # 베이스라인 핵심 계산 메서드들
    # ----------------------------------------------------------------
    def _calculate_transfer_time(self, task: Task, scene_id: int, provider: Provider) -> float:
        """파일 전송 시간 계산 (시간 단위)"""
        try:
            global_size = task.global_file_size
            scene_size = task.scene_size(scene_id)
            total_size = global_size + scene_size  # MB

            bandwidth = min(task.bandwidth, provider.bandwidth)  # MB/s
            if bandwidth <= 0:
                return float('inf')

            transfer_time_h = total_size / (bandwidth * 3600)  # 시간 단위로 변환
            return transfer_time_h
        except Exception:
            return float('inf')

    def _calculate_computation_time(self, task: Task, provider: Provider) -> float:
        """연산 시간 계산 (시간 단위)"""
        try:
            if provider.throughput <= 0:
                return float('inf')
            return task.scene_workload / provider.throughput
        except Exception:
            return float('inf')

    def _calculate_scene_total_time(self, task: Task, scene_id: int, provider: Provider) -> float:
        """단일 씬의 총 처리 시간 계산"""
        transfer_time = self._calculate_transfer_time(task, scene_id, provider)
        computation_time = self._calculate_computation_time(task, provider)
        return transfer_time + computation_time

    def _calculate_gpu_combination_metrics(self, task: Task, gpu_assignment: List[int], sim_time: datetime.datetime) -> \
    Tuple[float, float, bool, bool]:
        """
        GPU 조합의 총 시간, 총 비용, 예산/데드라인 만족 여부 계산
        같은 GPU에 배정된 씬들은 순차 처리, 서로 다른 GPU는 병렬 처리
        """
        try:
            # GPU별로 배정된 씬들을 그룹화
            gpu_scenes: Dict[int, List[int]] = {}
            for scene_id, provider_idx in enumerate(gpu_assignment):
                if provider_idx not in gpu_scenes:
                    gpu_scenes[provider_idx] = []
                gpu_scenes[provider_idx].append(scene_id)

            gpu_total_times = []
            total_cost = 0.0

            # 각 GPU별로 순차 처리 시간 계산
            for provider_idx, scenes_in_gpu in gpu_scenes.items():
                provider = self.providers[provider_idx]
                gpu_total_time = 0.0

                # 같은 GPU 내 씬들은 순차적으로 처리
                for scene_id in scenes_in_gpu:
                    scene_time = self._calculate_scene_total_time(task, scene_id, provider)
                    if scene_time == float('inf'):
                        return float('inf'), float('inf'), False, False

                    gpu_total_time += scene_time
                    scene_cost = scene_time * provider.price_per_gpu_hour
                    total_cost += scene_cost

                gpu_total_times.append(gpu_total_time)

            # T_tot = 가장 오래 걸리는 GPU의 총 시간 (GPU 간 병렬 처리)
            t_tot = max(gpu_total_times) if gpu_total_times else 0.0

            # 예산 체크
            budget_ok = total_cost <= task.budget

            # 데드라인 체크: sim_time 기준으로 T_tot 후 완료
            estimated_finish = sim_time + datetime.timedelta(hours=t_tot)
            deadline_ok = estimated_finish <= task.deadline

            return t_tot, total_cost, budget_ok, deadline_ok

        except Exception:
            return float('inf'), float('inf'), False, False

    def _calculate_efficiency(self, task: Task, t_tot: float, c_total: float) -> float:
        """효율성 계산: (scene_number × scene_workload) / (C_total × T_tot)"""
        try:
            if t_tot <= 0 or c_total <= 0:
                return 0.0

            total_workload = task.scene_number * task.scene_workload
            efficiency = total_workload / (c_total * t_tot)
            return efficiency
        except Exception:
            return 0.0

    def _find_best_gpu_combination(self, task: Task, sim_time: datetime.datetime) -> Optional[
        Tuple[float, List[int], float, float]]:
        """
        Task에 대해 최적의 GPU 조합 찾기 (조합 수 제한)
        Returns: (efficiency, gpu_assignment, t_tot, c_total) or None
        """
        best_efficiency = -1.0
        best_assignment = None
        best_metrics = None

        # GPU 중복 허용 조합 생성 (개수 제한)
        all_combinations = list(product(range(len(self.providers)), repeat=task.scene_number))
        limited_combinations = all_combinations[:self.max_combinations]

        if self.verbose:
            print(f"   조합 탐색: {len(limited_combinations)}/{len(all_combinations)}가지")

        for gpu_combination in limited_combinations:
            gpu_assignment = list(gpu_combination)

            # 1차: 시간/비용 계산 및 예산/데드라인 필터링
            t_tot, c_total, budget_ok, deadline_ok = self._calculate_gpu_combination_metrics(task, gpu_assignment,
                                                                                             sim_time)

            if not (budget_ok and deadline_ok):
                continue  # 조건 불만족 시 건너뛰기

            # 2차: 효율성 계산
            efficiency = self._calculate_efficiency(task, t_tot, c_total)

            # 최적 조합 선택
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_assignment = gpu_assignment
                best_metrics = (t_tot, c_total)

        if best_assignment is None:
            return None

        return (best_efficiency, best_assignment, best_metrics[0], best_metrics[1])

    # ----------------------------------------------------------------
    # 기존 Scheduler 인터페이스 호환 메서드들
    # ----------------------------------------------------------------
    def _feed(self, now: datetime.datetime, tasks: Tasks) -> None:
        """
        FIFO 방식: start_time <= now인 Task들을 대기 큐에 추가
        (기존 EDF 대신 FIFO 사용)
        """
        seen_task_ids = {task.id for task in self.waiting_tasks}

        for task in tasks:
            if task.start_time <= now and task.id not in seen_task_ids:
                # 아직 완전히 배정되지 않은 Task만 추가
                unassigned_scenes = [i for i, (start_time, _) in enumerate(task.scene_allocation_data)
                                     if start_time is None]

                if unassigned_scenes:
                    self.waiting_tasks.append(task)
                    seen_task_ids.add(task.id)

        # FIFO 순서: start_time 기준 정렬
        self.waiting_tasks.sort(key=lambda t: t.start_time)

    def _schedule_once(self, now: datetime.datetime, providers: Providers) -> List[Assignment]:
        """Task 단위로 최적 GPU 조합을 찾아 배정"""
        new_assignments: List[Assignment] = []
        remaining_tasks: List[Task] = []

        # providers를 인스턴스 변수로 저장 (다른 메서드에서 사용)
        self.providers = providers

        for task in self.waiting_tasks:
            # 이미 모든 씬이 배정된 Task는 건너뛰기
            unassigned_scenes = [i for i, (start_time, _) in enumerate(task.scene_allocation_data)
                                 if start_time is None]

            if not unassigned_scenes:
                continue  # 완전히 배정됨

            # 부분 배정된 Task는 일단 대기 큐에 유지 (향후 개선)
            if len(unassigned_scenes) != task.scene_number:
                remaining_tasks.append(task)
                continue

            if self.verbose:
                print(f"\n처리 중: {task.id} (씬 {task.scene_number}개)")

            # 최적 GPU 조합 찾기
            best_result = self._find_best_gpu_combination(task, now)

            if best_result is None:
                if self.verbose:
                    print(f"배정 실패: {task.id} - 조건을 만족하는 GPU 조합 없음")
                remaining_tasks.append(task)
                continue

            efficiency, gpu_assignment, t_tot, c_total = best_result
            if self.verbose:
                print(f"최적 조합: 효율성={efficiency:.4f}, 비용=${c_total:.2f}, 시간={t_tot:.2f}h")

            # Task를 선택된 GPU 조합에 배정
            task_assignments = self._assign_task_to_gpus(task, gpu_assignment, now)
            new_assignments.extend(task_assignments)

        self.waiting_tasks = remaining_tasks
        return new_assignments

    def _assign_task_to_gpus(self, task: Task, gpu_assignment: List[int], sim_time: datetime.datetime) -> List[
        Assignment]:
        """Task를 GPU 조합에 배정 (순차 처리 지원)"""
        assignments = []

        try:
            # GPU별로 배정된 씬들을 그룹화
            gpu_scenes: Dict[int, List[int]] = {}
            for scene_id, provider_idx in enumerate(gpu_assignment):
                if provider_idx not in gpu_scenes:
                    gpu_scenes[provider_idx] = []
                gpu_scenes[provider_idx].append(scene_id)

            # 각 GPU별로 순차적으로 씬들을 배정
            for provider_idx, scenes_in_gpu in gpu_scenes.items():
                provider = self.providers[provider_idx]
                current_start_time = sim_time

                for scene_id in scenes_in_gpu:
                    # 씬 처리 시간 계산
                    total_time = self._calculate_scene_total_time(task, scene_id, provider)

                    # 사용 가능한 시작 시간 확인
                    earliest_start = provider.earliest_available(total_time, current_start_time)

                    if earliest_start is None:
                        if self.verbose:
                            print(f"배정 실패: {task.id} Scene {scene_id} -> Provider {provider_idx} (시간 없음)")
                        continue

                    # Provider에 할당
                    provider.assign(task.id, scene_id, earliest_start, total_time)

                    # Task에 배정 정보 저장
                    task.scene_allocation_data[scene_id] = (earliest_start, provider_idx)

                    # 완료 시간 계산
                    finish_time = earliest_start + datetime.timedelta(hours=total_time)

                    # 결과에 추가
                    assignment = (task.id, scene_id, earliest_start, finish_time, provider_idx)
                    assignments.append(assignment)

                    if self.verbose:
                        print(f"배정 완료: {task.id} Scene {scene_id} -> Provider {provider_idx} "
                              f"({earliest_start.strftime('%m-%d %H:%M')} ~ {finish_time.strftime('%m-%d %H:%M')})")

                    # 다음 씬의 시작 시간을 현재 씬 완료 후로 설정
                    current_start_time = finish_time

        except Exception as e:
            if self.verbose:
                print(f"배정 중 오류 발생: {e}")

        return assignments

    # ----------------------------------------------------------------
    # 기존 Scheduler와 동일한 public entry-point
    # ----------------------------------------------------------------
    def run(
            self,
            tasks: Tasks,
            providers: Providers,
            time_start: Optional[datetime.datetime] = None,
            time_end: Optional[datetime.datetime] = None,
    ) -> List[Assignment]:
        """
        메인 시뮬레이션 루프 (기존 Scheduler 인터페이스 호환)
        """
        if time_start is None:
            earliest_task = min(t.start_time for t in tasks)
            earliest_prov = min(p.available_hours[0][0] for p in providers)
            time_start = min(earliest_task, earliest_prov)

        if time_end is None:
            time_end = max(t.deadline for t in tasks) + datetime.timedelta(days=1)

        if self.verbose:
            print(f"=== 베이스라인 스케줄러 시작 ===")
            print(f"시뮬레이션 기간: {time_start.strftime('%Y-%m-%d %H:%M')} ~ {time_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"최대 조합 수: {self.max_combinations}")

        now = time_start
        while now < time_end:
            if self.verbose:
                print(f"\n--- 현재 시간: {now.strftime('%Y-%m-%d %H:%M')} ---")

            self._feed(now, tasks)
            self.results.extend(self._schedule_once(now, providers))

            # 모든 씬이 배정되면 조기 종료
            if all(
                    all(st is not None for st, _ in t.scene_allocation_data) for t in tasks
            ):
                if self.verbose:
                    print(f"\n🎉 모든 씬 배정 완료!")
                break

            now += self.time_gap

        if self.verbose:
            print(f"\n=== 스케줄링 완료: 총 {len(self.results)}개 씬 배정 ===")
        return self.results


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    import json

    # Config 로드
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    tasks = Tasks()
    tasks.initialize_from_data(config['tasks'])

    providers = Providers()
    providers.initialize_from_data(config['providers'])

    # 스케줄러 실행
    scheduler = BaselineScheduler(max_combinations=81, verbose=False)  # 조합 수 증가, 조용한 모드
    results = scheduler.run(tasks, providers)

    # 결과 출력
    print(f"\n📋 베이스라인 스케줄러 결과:")
    print(f"총 배정된 씬: {len(results)}개")

    print(f"\n📊 배정 상세:")
    for assignment in results:
        task_id, scene_id, start, finish, provider_idx = assignment
        duration = (finish - start).total_seconds() / 3600
        print(f"  {task_id} Scene {scene_id}: Provider {provider_idx} "
              f"({start.strftime('%m-%d %H:%M')} ~ {finish.strftime('%m-%d %H:%M')}, {duration:.2f}h)")

    # 배정 통계
    total_scenes = sum(task.scene_number for task in tasks)
    assigned_scenes = len(results)
    print(f"\n📈 배정 통계:")
    print(f"전체 씬: {total_scenes}개")
    print(f"배정된 씬: {assigned_scenes}개 ({assigned_scenes / total_scenes * 100:.1f}%)")
    print(f"미배정 씬: {total_scenes - assigned_scenes}개 ({(total_scenes - assigned_scenes) / total_scenes * 100:.1f}%)")