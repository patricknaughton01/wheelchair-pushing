from planner import Planner
import time
import klampt


class Evaluator:
    def __init__(self, p: Planner, world_model: klampt.WorldModel):
        self.p = p
        self.world_model = world_model

    def eval(self) -> dict:
        stats = {}
        start = time.time()
        self.p.plan()
        stats["planning_time"] = time.time() - start
        while True:
            try:
                self.p.next()
            except StopIteration:
                break
