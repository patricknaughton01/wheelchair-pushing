import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Parse results file and "
        + "produce latex table output")
    parser.add_argument("paths", type=str, nargs="+", help="paths to results "
        + "files")
    args = vars(parser.parse_args())
    stat_keys = [
        "success",
        "planning_time",
        "execution_time",
        "trina_base_t_dist",
        "trina_strafe_dist",
        "trina_arm_cfg_dist",
        "wheelchair_t_dist",
        "wheelchair_t_accel_rmsa",
        "wheelchair_r_accel_rmsa"
    ]
    world_name_map = {
        "Model/worlds/world_long_turn.xml": "Long Turn",
        "Model/worlds/world_short_turn.xml": "Short Turn",
        "Model/worlds/world_corridor.xml": "Corridor",
        "Model/worlds/TRINA_world_cholera.xml": "Forward"
    }
    method_name_map = {
        "tracking": "HT",
        "tracking_cp": "HT (CP)",
        "tracking_sp": "HT (SP)",
        "state_lattice": "SL",
        "rrt": "KRRT"
    }
    world_order = [
        "Model/worlds/TRINA_world_cholera.xml",
        "Model/worlds/world_short_turn.xml",
        "Model/worlds/world_long_turn.xml",
        "Model/worlds/world_corridor.xml",
    ]
    method_order = ["tracking", "tracking_cp", "tracking_sp", "rrt", "state_lattice"]

    for p in args["paths"]:
        with open(p, "r") as f:
            res = json.load(f)
        print(f"============ {p} ============")
        world_strings = {}
        for method in res:
            print(f"------ {method} ------")
            for w in res[method]:
                for g in res[method][w]:
                    stats = res[method][w][g]
                    ret_str = ""
                    for s_key in stat_keys:
                        ret_str += f"& {float(stats[s_key]):.2f} "
                    if w not in world_strings:
                        world_strings[w] = {}
                    world_strings[w][method] = ret_str
                    print(ret_str)
        print("------ FULL TABLE ------")
        for i, w in enumerate(world_order):
            for j, method in enumerate(method_order):
                if not (w in world_strings and method in world_strings[w]):
                    continue
                val_str = world_strings[w][method]
                start_str = ""
                if j == 0:
                    start_str = f"\multirow{{5}}{{*}}{{{world_name_map[w]}}} "
                full_str = f"{start_str}& {method_name_map[method]} {val_str}\\\\"
                print(full_str)
            if i < len(world_order) - 1:
                print("\\midrule")

if __name__ == "__main__":
    main()
