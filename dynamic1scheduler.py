# dynamic_scheduler.py
import json
import datetime
from tasks import Tasks
from providers import Providers
from scheduler import Scheduler

def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def format_output(tasks: Tasks, assignments):
    # 각 씬에 배정된 정보 정리
    task_results = {task.properties.get("id"): [None] * task.scene_number for task in tasks}

    # assignments 형식: (task_id, provider_idx, scene_index, assign_time, total_duration)
    for (task_id, gpu_id, scene_id, start, duration) in assignments:
        finish = start + datetime.timedelta(hours=duration)
        task_results[task_id][scene_id] = (task_id, gpu_id, scene_id, start, finish, duration)

    # 씬이 배정되지 않은 경우 빈 리스트로
    final_output = []
    for task in tasks:
        task_id = task.properties.get("id")
        scenes = []
        for result in task_results[task_id]:
            if result is None:
                scenes.append([])
            else:
                task_id, gpu_id, scene_id, start, finish, duration = result
                scenes.append((task_id, gpu_id, scene_id, start, finish, duration))
        final_output.append(scenes)

    return final_output

def main():
    # Load config
    config = load_config("config.json")

    # Initialize providers and tasks
    tasks = Tasks()
    tasks.initialize_from_data(config["tasks"])

    providers = Providers()
    providers.initialize_from_data(config["providers"])

    # Run scheduler
    scheduler = Scheduler()
    assignments = scheduler.run(tasks, providers, datetime.timedelta(minutes=30))

    # Format and print result
    final_output = format_output(tasks, assignments)

    for i, task_scenes in enumerate(final_output):
        print(f"\n🧩 Task {i} 결과:")
        for s_idx, scene in enumerate(task_scenes):
            print(f"  - Scene {s_idx}:", scene)

if __name__ == "__main__":
    main()
