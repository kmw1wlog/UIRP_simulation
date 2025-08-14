import datetime as dt
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Model.tasks import Tasks
from Model.providers import Providers
from Core.Scheduler import system_evaluator
import pytest


def make_tasks_providers():
    # Two simple tasks, one provider
    tasks_data = [
        {
            "id": "T1",
            "scene_number": 1,
            "scene_file_size": 0,
            "global_file_size": 0,
            "scene_workload": 0,
            "bandwidth": 0,
            "budget": 1.5,
            "start_time": dt.datetime(2024, 1, 1, 0, 0),
            "deadline": dt.datetime(2024, 1, 1, 5, 0),
        },
        {
            "id": "T2",
            "scene_number": 1,
            "scene_file_size": 0,
            "global_file_size": 0,
            "scene_workload": 0,
            "bandwidth": 0,
            "budget": 1.0,
            "start_time": dt.datetime(2024, 1, 1, 0, 0),
            "deadline": dt.datetime(2024, 1, 1, 2, 0),
        },
    ]
    tasks = Tasks(); tasks.initialize_from_data(tasks_data)

    prov_data = [{
        "throughput": 1.0,
        "price": 1.0,
        "bandwidth": 1.0,
        "available_hours": []
    }]
    providers = Providers(); providers.initialize_from_data(prov_data)

    # Manually populate provider schedule
    start1 = dt.datetime(2024, 1, 1, 0, 0)
    end1 = dt.datetime(2024, 1, 1, 1, 0)
    start2 = dt.datetime(2024, 1, 1, 1, 0)
    end2 = dt.datetime(2024, 1, 1, 3, 0)
    providers[0].schedule = [
        ("T1", 0, start1, end1),
        ("T2", 0, start2, end2),
    ]
    return tasks, providers


def test_evaluate_metrics():
    tasks, providers = make_tasks_providers()
    metrics = system_evaluator.evaluate(tasks, providers)

    t1 = metrics["tasks"]["T1"]
    t2 = metrics["tasks"]["T2"]

    assert t1["cost"] == pytest.approx(1.0)
    assert t2["cost"] == pytest.approx(2.0)
    assert t2["budget_overrun"] == pytest.approx(1.0)

    assert metrics["makespan_hours"] == pytest.approx(3.0)
    assert metrics["throughput_tasks_per_hour"] == pytest.approx(2 / 3)
    assert metrics["deadline_hits"] == 1
    assert metrics["deadline_misses"] == 1
    assert metrics["average_lateness_hours"] == pytest.approx(0.5)
    assert metrics["provider_utilisation"][0] == pytest.approx(1.0)
