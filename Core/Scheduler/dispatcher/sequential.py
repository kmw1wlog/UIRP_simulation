# Core/Scheduler/dispatcher/sequential.py
import datetime
from typing import List, Dict
from Core.Scheduler.interface import Dispatcher

Assignment = tuple[str, int, datetime.datetime, datetime.datetime, int]

class SequentialDispatcher(Dispatcher):
    def _ea(self, prov, dur, cur):
        try:
            return prov.earliest_available(dur, cur)
        except TypeError:
            return prov.earliest_available(dur)

    def dispatch(self, t, cmb, now, ps, ev, verbose):
        out: List[Assignment] = []
        groups: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            if p == -1:
                continue  # 이번 스텝 미배치
            groups.setdefault(p, []).append(sid)

        for p, sids in groups.items():
            prov, cur = ps[p], now
            for sid in sids:
                # 이미 배정된 씬은 스킵
                if t.scene_allocation_data[sid][0] is not None:
                    continue
                dur, _ = ev.time_cost(t, sid, prov)
                st = self._ea(prov, dur, cur)
                if st is None:
                    continue
                prov.assign(t.id, sid, st, dur)
                t.scene_allocation_data[sid] = (st, p)
                ft = st + datetime.timedelta(hours=dur)
                out.append((t.id, sid, st, ft, p))
                if verbose:
                    print(f"      scene{sid}->{p} {st.strftime('%m-%d %H:%M')} dur={dur:.2f}")
                cur = ft
        return out

class CPSatDispatcher(SequentialDispatcher):
    pass
