import argparse
import os
from pathlib import Path
import time
from threading import Thread

import klampt
import numpy as np
import cv2
from klampt import vis
from klampt.vis.glprogram import GLViewport
from planner import Planner

from tracking_planner import TrackingPlannerInstance

running_flag = True
sim_running_flag = True


def main():
    global running_flag, sim_running_flag
    parser = argparse.ArgumentParser(description="Test the tracking planner")
    parser.add_argument("type", type=str, choices=["tracking"],
        help="which planner to use")
    parser.add_argument("-s", action="store_true",
        help="save frames from the simulation")
    args = vars(parser.parse_args())
    world_fn = "Model/worlds/world_short_turn.xml"
    world = klampt.WorldModel()
    world.loadFile(world_fn)
    dt = 1 / 50
    if args["type"] == "tracking":
        planner = TrackingPlannerInstance(world_fn, dt)
    else:
        raise ValueError("Unknown planner type")
    planner.plan(np.array([0.0, -10.0, 0.0]), 0.5, 0.5)
    iter = 0
    vis.add("world", world)
    viewport = GLViewport()
    viewport.load_file("vis/example_traj.txt")
    vis.setViewport(viewport)
    vis.show()
    fps = 30
    vis_dt = 1 / fps
    save_dir = "tmp"
    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(exist_ok=True, parents=True)
    thread = Thread(target=advance_sim, args=(world, planner, dt), daemon=True)
    thread.start()
    while vis.shown():
        start = time.monotonic()
        if sim_running_flag and args["s"]:
            img = vis.screenshot()
            cv2.imwrite(os.path.join(save_dir, f'img_{iter:05}.jpg'),
                np.flip(img, 2))
            iter += 1
        time.sleep(max(0.001, vis_dt - (time.monotonic() - start)))
    running_flag = False
    thread.join()


def advance_sim(world: klampt.WorldModel, planner: Planner, dt: float):
    global running_flag, sim_running_flag
    robot_model = world.robot("trina")
    wheelchair_model = world.robot("wheelchair")
    iter = 0
    while running_flag:
        iter += 1
        try:
            planner.next()
            robot_model.setConfig(planner.robot_model.getConfig())
            wheelchair_model.setConfig(planner.wheelchair_model.getConfig())
            time.sleep(dt)
        except StopIteration:
            print("Stopped at iteration ", iter)
            break
    sim_running_flag = False


if __name__ == "__main__":
    main()
