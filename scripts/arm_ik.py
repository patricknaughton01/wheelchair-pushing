import json
import argparse
import os
import klampt
from klampt.model import ik
import sys
sys.path.append("../")
from consts import SETTINGS_PATH

def main():
    with open(os.path.join("../", SETTINGS_PATH), "r") as f:
        settings = json.load(f)
    parser = argparse.ArgumentParser(description="Get arm joint angles "
        + "that achieve a desired tool link pose")
    parser.add_argument("pose", nargs=12, type=float, help="Pose to "
        + "achieve, column major rotation matrix followed by "
        + "translation vector, should be 12 numbers.")
    parser.add_argument("-s", type=str, default="left",
        choices=["left", "right"], help="Which arm to solve for.")
    parser.add_argument("-i", type=float, nargs=6, default=[0,0,0,0,0,0],
        help="Initial joint configuration guess, 6 numbers if provided.")
    args = vars(parser.parse_args())
    world = klampt.WorldModel()
    robot_model: klampt.RobotModel = world.loadRobot(
        os.path.join("../", settings["robot_fn"]))
    q = robot_model.getConfig()
    side = args["s"]
    dofs = settings["{}_arm_dofs".format(side)]
    for i, d in enumerate(dofs):
        q[d] = args["i"][i]
    robot_model.setConfig(q)
    R = args["pose"][:9]
    p = args["pose"][9:]
    goal = ik.objective(
        robot_model.link("{}_tool_link".format(side)), R=R, t=p)
    if ik.solve_global(goal, activeDofs=settings["{}_arm_dofs".format(side)]):
        q = robot_model.getConfig()
        out = []
        for d in dofs:
            out.append(q[d])
        print("Joint angles: ", out)
    else:
        print("IK failed")


if __name__ == "__main__":
    main()
