import time
import json
import klampt
from klampt import vis
from consts import SETTINGS_PATH


def main():
    world = klampt.WorldModel()
    world.loadFile("Model/worlds/TRINA_world_cholera.xml")
    robot_model: klampt.RobotModel = world.robot(0)
    q = robot_model.getConfig()
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
    # t_cfg = [-1.2396209875689905, -2.6281658611693324, -1.0119028091430664, 5.456587779312887, -2.3149259726153772, 0.15132474899291992]
    # t_cfg = [-4.635901893690355, 5.806223419961121, 1.082262209315542, -2.753160794116381, 1.0042225011740618, 4.022876408436895]
    t_cfg = [0.17317459, -1.66203799, -2.25021315, 3.95050542, -0.59267456, -0.8280866]
    for i, d in enumerate(settings["left_arm_dofs"]):
        q[d] = t_cfg[i]
    robot_model.setConfig(q)
    print(robot_model.link("left_tool_link").getTransform())
    vis.add("world", world)
    vis.show()
    while vis.shown():
        time.sleep(0.01)
    vis.kill()


if __name__ == "__main__":
    main()
