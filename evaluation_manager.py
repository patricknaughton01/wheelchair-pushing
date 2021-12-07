import argparse
import json
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import time
from tqdm import tqdm

import numpy as np
from evaluator import Evaluator
from tracking_planner import TrackingPlannerInstance
from state_lattice_planner import StateLatticePlanner, sl
from decoupled_rrt_planner import DecoupledRRTPlanner

class EvaluationManager:
    def __init__(self, p_name: str, world_fn: str, goal: np.ndarray, cfg_fn: str):
        self.cfg_fn = cfg_fn
        with open(self.cfg_fn, "r") as f:
            self.eval_settings = json.load(f)
        self.p_name = p_name
        self.world_fn = world_fn
        self.goal = goal
        self.p_conn, self.c_conn = Pipe()
        self.eval_proc = None

    def run_eval(self) -> dict:
        self.eval_proc = Process(target=self.eval,
            args=(self.p_name, self.world_fn, self.goal, self.c_conn),
            daemon=True)
        state = "planning"
        status = None
        self.eval_proc.start()
        self.p_conn.recv()
        self.p_conn.send(0) # Signal to start planning
        start = time.monotonic()
        while True:
            if state == "planning":
                if self.p_conn.poll():
                    if self.p_conn.recv() == 1:
                        state = "executing"
                        self.p_conn.send(0) # Signal to start executing
                        start = time.monotonic()
                    else:
                        status = "planning_failed"
                        break
                elif time.monotonic() - start > self.eval_settings["plan_timeout"]:
                    status = "planning_failed"
                    break
            elif state == "executing":
                if self.p_conn.poll():
                    if self.p_conn.recv() == 2:
                        state = "evaluating"
                        break
                    else:
                        status = "executing_failed"
                        break
                elif time.monotonic() - start > self.eval_settings["exec_timeout"]:
                    status = "executing_failed"
                    break
            time.sleep(self.eval_settings["manager_dt"])
        if status is None:
            # Evaluating trajectory, return the stats once completed
            ret = self.p_conn.recv()
        else:
            # Error occured, return reason
            ret = {"error": status}
        # Clean up the process
        if self.eval_proc.is_alive():
            self.eval_proc.terminate()
        self.eval_proc.join()
        if "error" not in ret and (ret.get("execution_time", float('inf')) > self.eval_settings["exec_timeout"]):
            ret = {"error": "executing_failed"}
        return ret

    def eval(self, p_name: str, world_fn: str, goal: np.ndarray, c_conn: Connection):
        # TODO: Support other planner keys
        if p_name == "tracking":
            p = TrackingPlannerInstance(world_fn, self.eval_settings["dt"])
        elif p_name == "state_lattice":
            p = StateLatticePlanner(sl, world_fn, self.eval_settings["dt"])
        elif p_name == "rrt":
            p = DecoupledRRTPlanner(world_fn, self.eval_settings["dt"])
        else:
            raise ValueError("Didn't recognize planner name: %s", p_name)
        e = Evaluator(p, world_fn)
        c_conn.send(0)  # Ready to plan
        c_conn.recv()   # Wait to start planning
        e.eval_plan(goal, self.eval_settings["disp_tol"],
            np.radians(self.eval_settings["rot_tol"]))
        c_conn.send(1)  # Done planning, ready to execute
        c_conn.recv()   # Wait to start executing
        e.eval_exec()
        c_conn.send(2)  # Done executing
        e.eval_traj()
        c_conn.send(e.stats)


def main():
    parser = argparse.ArgumentParser(description="Run evaluations of the "
        + "planners")
    parser.add_argument("-c", type=str, default="eval_config.json",
        help="eval config file name")
    parser.add_argument("-o", type=str, default="res.json",
        help="output file name")
    args = vars(parser.parse_args())
    with open(args["c"], "r") as f:
        eval_settings = json.load(f)
    res = {}
    total_settings = []
    for p_name in eval_settings["planners"]:
        res[p_name] = {}
        num_itr = 10
        if p_name == "rrt":
            for i in range(num_itr):
                for test in eval_settings["tests"]:
                    world_fn = test["world_fn"]
                    res[p_name][world_fn] = {}
                    for g in test["goals"]:
                        total_settings.append((p_name, world_fn, g))
        else:
            for test in eval_settings["tests"]:
                world_fn = test["world_fn"]
                res[p_name][world_fn] = {}
                for g in test["goals"]:
                    total_settings.append((p_name, world_fn, g))

    for setting in tqdm(total_settings):
        print("Working on setting: ")
        print(setting)
        p_name, world_fn, g = setting
        goal = np.array(g, dtype="float64")
        goal[2] = np.radians(goal[2])
        m = EvaluationManager(p_name, world_fn, goal, args["c"])
        # Eventually dump to JSON, all keys need to be strs

        if p_name == "rrt":
            result = m.run_eval()

            if result["success"] == True:
                result["success"] = 1

            elif result["success"] == False:
                result["success"] = 0

            if res[p_name][world_fn] == {}:
                res[p_name][world_fn][str(g)] = result
            
            else:
                if result["success"] == 1:
                    for metric in result.keys():
                        res[p_name][world_fn][str(g)][metric] += result[metric]

        else:
            res[p_name][world_fn][str(g)] = m.run_eval()
    
    for world in res["rrt"]:
        for goal in res["rrt"][world]:
            success_num = res["rrt"][world][goal]["success"]
            print(success_num)
            if success_num != 0:
                for metric in res["rrt"][world][goal]:
                    if metric == "success":
                        res["rrt"][world][goal][metric] = res["rrt"][world][goal][metric]/num_itr
                    else: 
                        res["rrt"][world][goal][metric] = res["rrt"][world][goal][metric]/success_num

    with open(args["o"], "w") as f:
        json.dump(res, f, indent="\t")


if __name__ == "__main__":
    main()
