from multiprocessing import Process
from threading import Thread


class EvaluationManager:
    def __init__(self, p: Planner, world_fn: str):
        self.eval_proc = Process(target=self.eval, args=(p, world_fn))

    def eval(self, p: Planner, world_fn: str):
        pass
