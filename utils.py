from __future__ import annotations
import datetime
from typing import List, Tuple

def merge_intervals(iv: List[Tuple[datetime.datetime, datetime.datetime]]):
    if not iv:
        return []
    iv.sort(key=lambda t: t[0])
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        m_s, m_e = merged[-1]
        if s <= m_e:
            merged[-1][1] = max(m_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

if __name__ == "__main__":
    import datetime as dt

    tests = [
        (
            "overlap_simple",
            [
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 10)),
                (dt.datetime(2025, 1, 1, 9, 30), dt.datetime(2025, 1, 1, 11)),
            ],
            [
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 11)),
            ],
        ),
        (
            "touching_edges",
            [
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 10)),
                (dt.datetime(2025, 1, 1, 10), dt.datetime(2025, 1, 1, 11)),
            ],
            [
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 11)),
            ],
        ),
        (
            "disjoint",
            [
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 10)),
                (dt.datetime(2025, 1, 1, 11), dt.datetime(2025, 1, 1, 12)),
            ],
            [
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 10)),
                (dt.datetime(2025, 1, 1, 11), dt.datetime(2025, 1, 1, 12)),
            ],
        ),
        (
            "unsorted_input",
            [
                (dt.datetime(2025, 1, 1, 11), dt.datetime(2025, 1, 1, 12)),
                (dt.datetime(2025, 1, 1, 9),  dt.datetime(2025, 1, 1, 10, 30)),
                (dt.datetime(2025, 1, 1, 10), dt.datetime(2025, 1, 1, 10, 45)),
            ],
            [
                (dt.datetime(2025, 1, 1, 9), dt.datetime(2025, 1, 1, 10, 45)),
                (dt.datetime(2025, 1, 1, 11), dt.datetime(2025, 1, 1, 12)),
            ],
        ),
        ("empty", [], []),
    ]

    for name, data, expected in tests:
        result = merge_intervals(data.copy())
        assert result == expected, f"{name} failed: {result} != {expected}"
    print("all tests passed")
