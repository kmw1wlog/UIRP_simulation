import json
import datetime
from typing import List, Tuple, Optional
from tasks import Tasks, Task
from providers import Providers, Provider

# 출력 형식: (task_id, scene_id, start_time, finish_time, provider_idx)
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]


class Scheduler_baseline:
    """
    동적 스케줄러 (작업배정기)
    전략: 효율성 = throughput / (price * time) 값이 큰 GPU에 
         workload가 큰 Task 순으로 1대1 매칭
    """
    
    def __init__(self):
        self.tasks = Tasks()
        self.providers = Providers()
        self.results: List[Assignment] = []
    
    def load_config(self, config_path: str = "config.json"):
        """config.json에서 테스팅 데이터 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.tasks.initialize_from_data(config['tasks'])
        self.providers.initialize_from_data(config['providers'])
    
    def calculate_efficiency(self, task: Task, provider: Provider) -> float:
        """
        효율성 계산: throughput / price (단순화)
        """
        efficiency = provider.throughput / provider.price_per_gpu_hour
        return efficiency
    
    def calculate_total_time(self, task: Task, provider: Provider) -> float:
        """
        총 소요 시간 계산 (단순화: 연산 시간만 고려)
        """
        # 연산 시간만 고려
        computation_time = task.scene_workload / provider.throughput
        return computation_time
    
    def earliest_available_sim(self, provider: Provider, duration_h: float, sim_time: datetime.datetime) -> Optional[datetime.datetime]:
        """
        시뮬레이션 시간 기준으로 가장 빠른 사용 가능한 시작 시간 반환
        """
        for start, end in sorted(provider.available_hours, key=lambda t: t[0]):
            if end <= sim_time:
                continue

            check_start = max(sim_time, start)
            check_end = end
            
            # 예약된 시간 슬롯 확인
            busy_slots = sorted([(s, f) for _, _, s, f in provider.schedule if s < check_end and f > check_start])

            candidate_start = check_start
            for res_start, res_end in busy_slots:
                gap_h = (res_start - candidate_start).total_seconds() / 3600
                if gap_h >= duration_h:
                    return candidate_start
                candidate_start = max(candidate_start, res_end)

            if (check_end - candidate_start).total_seconds() / 3600 >= duration_h:
                return candidate_start

        return None
    
    def feed(self, event=None):
        """
        이벤트 발생 시 tasks와 providers 업데이트
        현재는 config.json에서 정적 로드만 구현
        """
        pass
    
    def schedule(self, sim_time: datetime.datetime) -> List[Assignment]:
        """
        할당 가능한 task를 전략에 따라 매칭
        workload가 큰 task 순으로 정렬 후, 효율성 높은 provider에 배정
        """
        assignments = []
        
        # workload 기준으로 task 정렬 (큰 순서대로)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.scene_workload, reverse=True)
        
        for task in sorted_tasks:
            # task 시작 시간이 되었는지 확인
            if task.start_time > sim_time:
                continue
                
            for scene_id in range(task.scene_number):
                # 이미 배정된 scene은 건너뛰기
                if task.scene_allocation_data[scene_id][0] is not None:
                    continue
                
                # 각 provider의 효율성 계산
                provider_efficiency = []
                for idx, provider in enumerate(self.providers):
                    total_time = self.calculate_total_time(task, provider)
                    earliest_start = self.earliest_available_sim(provider, total_time, sim_time)
                    
                    if earliest_start is not None:
                        efficiency = self.calculate_efficiency(task, provider)
                        provider_efficiency.append((efficiency, idx, earliest_start, total_time))
                
                # 효율성 높은 순으로 정렬
                provider_efficiency.sort(key=lambda x: x[0], reverse=True)
                
                # 가장 효율성 높은 provider에 배정
                if provider_efficiency:
                    efficiency, provider_idx, start_time, duration = provider_efficiency[0]
                    provider = self.providers[provider_idx]
                    
                    # Provider에 할당
                    provider.assign(task.id, scene_id, start_time, duration)
                    
                    # Task에 배정 정보 저장
                    task.scene_allocation_data[scene_id] = (start_time, provider_idx)
                    
                    # 완료 시간 계산
                    finish_time = start_time + datetime.timedelta(hours=duration)
                    
                    # 결과에 추가
                    assignment = (task.id, scene_id, start_time, finish_time, provider_idx)
                    assignments.append(assignment)
                    
                    print(f"배정: {task.id} Scene {scene_id} -> Provider {provider_idx} ({start_time.strftime('%m-%d %H:%M')} ~ {finish_time.strftime('%m-%d %H:%M')})")
        
        return assignments
    
    def run(self, time_gap: datetime.timedelta = datetime.timedelta(hours=1)):
        """
        메인 실행 루프
        시뮬레이션 시간을 기준으로 스케줄링 수행
        """
        # config.json 데이터 로드
        self.load_config()
        
        # 시뮬레이션 시작 시간: 가장 이른 task 시작 시간
        sim_time = min(task.start_time for task in self.tasks)
        
        # 시뮬레이션 종료 시간: 가장 늦은 deadline + 1일
        end_time = max(task.deadline for task in self.tasks) + datetime.timedelta(days=1)
        
        print(f"시뮬레이션 시작: {sim_time.strftime('%Y-%m-%d %H:%M')}")
        
        # 시뮬레이션 루프
        while sim_time < end_time:
            # 이벤트 피드 (현재는 정적)
            self.feed()
            
            # 스케줄링 수행
            assignments = self.schedule(sim_time)
            self.results.extend(assignments)
            
            # 시간 전진
            sim_time += time_gap
            
            # 모든 scene이 배정되면 조기 종료
            all_assigned = all(
                all(allocation[0] is not None for allocation in task.scene_allocation_data)
                for task in self.tasks
            )
            if all_assigned:
                print(f"모든 scene 배정 완료!")
                break
        
        return self.results
    
    def print_results(self):
        """결과를 보기 좋게 출력"""
        print("\n=== 스케줄링 결과 ===")
        
        # Task별로 그룹화
        task_groups = {}
        for assignment in self.results:
            task_id, scene_id, start_time, finish_time, provider_idx = assignment
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(assignment)
        
        for task_id, assignments in task_groups.items():
            print(f"\n▶ {task_id}:")
            for assignment in sorted(assignments, key=lambda x: x[1]):  # scene_id로 정렬
                task_id, scene_id, start_time, finish_time, provider_idx = assignment
                print(f"  Scene {scene_id}: Provider {provider_idx}")
                print(f"    시작: {start_time}")
                print(f"    종료: {finish_time}")
                print(f"    소요시간: {finish_time - start_time}")


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    # 스케줄러 생성 및 실행
    scheduler = Scheduler_baseline()
    results = scheduler.run()
    
    print(f"\n총 {len(results)}개 scene 배정 완료!")
    print("\n출력 형식 (task_id, scene_id, start_time, finish_time, provider_idx):")
    for assignment in results:
        task_id, scene_id, start_time, finish_time, provider_idx = assignment
        print(f"  {assignment}")
    
    # 결과 상세 출력
    scheduler.print_results()
    
    # 효율성 요약
    print(f"\n=== 효율성 요약 ===")
    for idx, provider in enumerate(scheduler.providers):
        efficiency = provider.throughput / provider.price_per_gpu_hour
        scenes_assigned = len([a for a in results if a[4] == idx])
        print(f"Provider {idx}: 효율성={efficiency:.2f}, 배정된 scene={scenes_assigned}개") 