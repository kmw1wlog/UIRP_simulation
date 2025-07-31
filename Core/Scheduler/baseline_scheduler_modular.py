# ================================================================
# baseline_scheduler_modular.py
# ================================================================
from __future__ import annotations
import datetime, itertools, math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Sequence

from Model.tasks import Tasks, Task
from Model.providers import Providers, Provider

Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

# ----------------------------------------------------------------
# 1) 전략 인터페이스
# ----------------------------------------------------------------
class TaskSelector(ABC):
    @abstractmethod
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]: ...

class ComboGenerator(ABC):
    @abstractmethod
    def best_combo(
        self, task: Task, providers: Providers, sim_time: datetime.datetime,
        evaluator: "MetricEvaluator", verbose: bool = False,
    ) -> Optional[Tuple[List[int], float, float]]: ...

class MetricEvaluator(ABC):
    @abstractmethod
    def time_cost(self, task: Task, scene_id: int, prov: Provider) -> Tuple[float, float]: ...
    @abstractmethod
    def feasible(self, task: Task, combo: List[int], sim_time: datetime.datetime,
                 providers: Providers) -> Tuple[bool, float, float]: ...
    @abstractmethod
    def efficiency(self, task: Task, t_tot: float, cost: float) -> float: ...

class Dispatcher(ABC):
    @abstractmethod
    def dispatch(
        self, task: Task, combo: List[int], sim_time: datetime.datetime,
        providers: Providers, evaluator: MetricEvaluator, verbose: bool
    ) -> List[Assignment]: ...

# ----------------------------------------------------------------
# 2) 기본 전략
# ----------------------------------------------------------------
class FIFOTaskSelector(TaskSelector):
    def select(self, now, waiting): return list(waiting)



class BaselineEvaluator(MetricEvaluator):
    def __init__(self): self._c: Dict[tuple, float] = {}
    def _t(self, t,s,p):
        k=("tx",t.id,s,p);                                                     # transfer
        if k in self._c: return self._c[k]
        v=float('inf') if (bw:=min(t.bandwidth,p.bandwidth))<=0 else \
           (t.global_file_size+t.scene_size(s))/bw/3600
        self._c[k]=v; return v
    def _cpt(self,t,p):
        k=("cmp",t.id,p)
        if k in self._c: return self._c[k]
        v=float('inf') if p.throughput<=0 else t.scene_workload/p.throughput
        self._c[k]=v; return v
    def time_cost(self,t,s,p):
        d=self._t(t,s,p)+self._cpt(t,p); return d,d*p.price_per_gpu_hour
    def feasible(self,t,cmb,now,ps):
        grouped:Dict[int,List[int]]={}
        for sid,p in enumerate(cmb): grouped.setdefault(p,[]).append(sid)
        t_tot,cost=0.0,0.0
        for p,sids in grouped.items():
            seq=sum(self.time_cost(t,sid,ps[p])[0] for sid in sids)
            if any(math.isinf(self.time_cost(t,sid,ps[p])[0]) for sid in sids): return False,math.inf,math.inf
            t_tot=max(t_tot,seq)
            cost += sum(self.time_cost(t,sid,ps[p])[1] for sid in sids)
        if cost>t.budget: return False,t_tot,cost
        if now+datetime.timedelta(hours=t_tot)>t.deadline: return False,t_tot,cost
        return True,t_tot,cost
    def efficiency(self,t,t_tot,cost):
        return 0.0 if t_tot<=0 or cost<=0 else (t.scene_number*t.scene_workload)/(cost*t_tot)

class BruteForceGenerator(ComboGenerator):
    def __init__(self,max_comb=81): self.max_comb=max_comb
    def best_combo(self,t,ps,now,ev,verbose=False):
        best=(-1.0,None)
        for cmb in itertools.islice(itertools.product(range(len(ps)),repeat=t.scene_number),self.max_comb):
            ok,t_tot,cost=ev.feasible(t,list(cmb),now,ps)
            if verbose: print(f"    ↪ {cmb} ok={ok} t={t_tot:.2f}h cost={cost:.1f}$")
            if not ok: continue
            eff=ev.efficiency(t,t_tot,cost)
            if eff>best[0]: best=(eff,(list(cmb),t_tot,cost))
        return None if best[1] is None else best[1]

class SequentialDispatcher(Dispatcher):
    def dispatch(self,t,cmb,now,ps,ev,verbose):
        out:List[Assignment]=[]; groups:Dict[int,List[int]]={}
        for sid,p in enumerate(cmb): groups.setdefault(p,[]).append(sid)
        for p,sids in groups.items():
            prov,cur=ps[p],now
            for sid in sids:
                dur,_=ev.time_cost(t,sid,prov); st=prov.earliest_available(dur,cur)
                if st is None: continue
                prov.assign(t.id,sid,st,dur)
                t.scene_allocation_data[sid]=(st,p)
                ft=st+datetime.timedelta(hours=dur)
                out.append((t.id,sid,st,ft,p))
                if verbose: print(f"      scene{sid}->{p} {st.strftime('%m-%d %H:%M')} dur={dur:.2f}")
                cur=ft
        return out

# ----------------------------------------------------------------
# 3) OR-Tools 전략 (선택)
# ----------------------------------------------------------------
try:
    from ortools.sat.python import cp_model
    class CPSatComboGenerator(ComboGenerator):
        def best_combo(self,t,ps,now,ev,verbose=False):
            m=cp_model.CpModel();S,P=t.scene_number,len(ps)
            x=[[m.NewBoolVar(f"x{s}_{p}") for p in range(P)] for s in range(S)]
            for s in range(S): m.Add(sum(x[s])==1)

            big=10**7; dur=[[0]*P for _ in range(S)]; cost=[[0]*P for _ in range(S)]
            for s in range(S):
                for p in range(P):
                    d,c=ev.time_cost(t,s,ps[p])
                    dur[s][p]=big if math.isinf(d) else int(d*3600)
                    cost[s][p]=big if math.isinf(d) else int(c*100)
            makespan=m.NewIntVar(0,big,"mk")
            for p in range(P):
                tot=m.NewIntVar(0,big,f"tot{p}")
                m.Add(tot==sum(dur[s][p]*x[s][p] for s in range(S))); m.Add(tot<=makespan)
            total_cost=sum(cost[s][p]*x[s][p] for s in range(S) for p in range(P))
            m.Add(total_cost<=int(t.budget*100))
            m.Add(makespan<=int((t.deadline-now).total_seconds()))
            m.Minimize(makespan)

            slv=cp_model.CpSolver()
            if verbose: slv.parameters.log_search_progress=True
            slv.parameters.max_time_in_seconds=5
            st=slv.Solve(m)
            if st not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None
            cmb=[next(p for p in range(P) if slv.BooleanValue(x[s][p])) for s in range(S)]
            return cmb,slv.Value(makespan)/3600,slv.Value(total_cost)/100
    class CPSatDispatcher(SequentialDispatcher): pass
except ImportError:
    CPSatComboGenerator=CPSatDispatcher=None

# ----------------------------------------------------------------
# 4) 레지스트리 & Scheduler
# ----------------------------------------------------------------
COMBO_REG={"bf":BruteForceGenerator}
DISP_REG ={"bf":SequentialDispatcher}
if CPSatComboGenerator: COMBO_REG["cp"]=CPSatComboGenerator; DISP_REG["cp"]=CPSatDispatcher

class BaselineScheduler:
    def __init__(self,*,algo="bf",time_gap=datetime.timedelta(hours=1),
                 selector:TaskSelector=FIFOTaskSelector(),
                 evaluator:MetricEvaluator=BaselineEvaluator(),
                 verbose=False):
        self.selector=selector
        self.generator=COMBO_REG[algo]()
        self.dispatcher=DISP_REG[algo]()
        self.evaluator=evaluator
        self.time_gap=time_gap
        self.verbose=verbose
        self.waiting_tasks:List[Task]=[]
        self.results:List[Assignment]=[]

    def _feed(self,now,tasks):
        ids={t.id for t in self.waiting_tasks}
        for t in tasks:
            if t.start_time<=now and t.id not in ids and any(st is None for st,_ in t.scene_allocation_data):
                self.waiting_tasks.append(t)
        self.waiting_tasks.sort(key=lambda t:t.start_time)

    def _schedule_once(self,now,ps):
        new:List[Assignment]=[]; remain=[]
        for t in self.selector.select(now,self.waiting_tasks):
            if not all(st is None for st,_ in t.scene_allocation_data):
                remain.append(t); continue
            best=self.generator.best_combo(t,ps,now,self.evaluator,
                                           verbose=self.verbose)
            if best is None: remain.append(t); continue
            cmb,t_tot,cost=best
            if self.verbose: print(f"[{t.id}] choose {cmb} t={t_tot:.2f}h cost={cost:.1f}$")
            new+=self.dispatcher.dispatch(t,cmb,now,ps,self.evaluator,self.verbose)
        self.waiting_tasks=remain; return new

    def run(self,tasks:Tasks,ps:Providers,
            time_start:datetime.datetime|None=None,
            time_end:datetime.datetime|None=None)->List[Assignment]:
        if time_start is None:
            time_start=min(min(p.available_hours)[0] for p in ps)
        now=time_start
        if time_end is None:
            time_end=max(t.deadline for t in tasks)+datetime.timedelta(days=1)
        while now<time_end:
            self._feed(now,tasks)
            self.results+=self._schedule_once(now,ps)
            if all(all(st is not None for st,_ in t.scene_allocation_data) for t in tasks):
                break
            now+=self.time_gap
        return self.results
