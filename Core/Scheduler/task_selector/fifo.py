from Core.Scheduler.interface import TaskSelector

class FIFOTaskSelector(TaskSelector):
    def select(self, now, waiting):
        return list(waiting)