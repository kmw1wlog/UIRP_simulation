import datetime
from typing import List, Dict
from Core.Scheduler.interface import Dispatcher

Assignment = tuple[str, int, datetime.datetime, datetime.datetime, int]

class SequentialDispatcher(Dispatcher):
    def dispatch(self, t, cmb, now, ps, ev, verbose):
        out: List[Assignment] = []
        groups: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            groups.setdefault(p, []).append(sid)

        for p, sids in groups.items():
            prov, cur = ps[p], now
            for sid in sids:
                dur, _ = ev.time_cost(t, sid, prov)
                st = prov.earliest_available(dur, cur)
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
