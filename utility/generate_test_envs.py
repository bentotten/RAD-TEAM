import gym
import numpy as np
import joblib
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import visilibity as vis
from gym.utils.seeding import _int_list_from_bigint, hash_seed


EPSILON = 0.0000001


def create_envs(num_envs, init_dims, env_name, save_path):
    env_dict = {}
    for ii in range(num_envs):
        env = gym.make(env_name, **init_dims)
        env.reset()
        if init_dims["obstruct"] > 0 or init_dims["obstruct"] == -1:
            env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, env.bkg_intensity, env.obs_coord)
        else:
            env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, env.bkg_intensity)
            print(f"Source coord: {env.src_coords}, Det coord: {env.agents[0].det_coords}, Intensity: {env.intensity},{env.bkg_intensity}")

    joblib.dump(env_dict, osp.join(save_path, "test_env_dict_obs" + str(init_dims["obstruct"])))


def create_envs_snr(num_envs, init_dims, env_name, save_path, split=4, snr="low"):
    env_dict = {}
    ii = 0
    snr_range = {"none": [0, 0], "low": [1.0, 1.2], "med": [1.2, 1.6], "high": [1.6, 2.0]}
    div = np.round((snr_range[snr][1] - snr_range[snr][0]) / (split), 2)
    num_envs_split = round(num_envs / (split))
    counts = np.zeros(split)

    while ii < num_envs:
        env = gym.make(env_name, **init_dims)
        env.reset()
        det = np.linalg.norm(env.src_coords - np.array(env.agents[0].det_coords))  # NOTE: Agents begin in same location
        meas = env.intensity / (det**2) + env.bkg_intensity
        if snr == "none":
            if init_dims["obstruction_count"] > 0 or init_dims["obstruction_count"] == -1:
                env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, snr_range["none"][0], env.obs_coord)
                ii += 1
            else:
                env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, snr_range["none"][0])
                ii += 1
        else:
            snr_exp = meas / env.bkg_intensity
            if snr_range[snr][0] < snr_exp <= snr_range[snr][1]:
                if snr == "med" or snr == "low" or snr == "high":
                    counts, inc_flag = classify_snr(np.round(snr_exp, 3), div, counts, num_envs_split, lb=snr_range[snr][0])
                    if init_dims["obstruction_count"] > 0 or init_dims["obstruction_count"] == -1:
                        if inc_flag:
                            env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, env.bkg_intensity, env.obs_coord)
                            ii += 1
                            if (ii % 100) == 0:
                                print(f"Obs SNR: {np.round(snr_exp,3)} -> {counts}")
                    else:
                        if inc_flag:
                            env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, env.bkg_intensity)
                            ii += 1
                            if (ii % 100) == 0:
                                print(f"SNR: {np.round(snr_exp,3)} -> {counts}")
                else:
                    env_dict["env_" + str(ii)] = (env.src_coords, env.agents[0].det_coords, env.intensity, env.bkg_intensity)
                    ii += 1
                    print(f"SNR: {np.round(snr_exp,3)}")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    joblib.dump(env_dict, osp.join(
        save_path,
        f"test_env_obs{init_dims['obstruction_count']}_{snr}_{init_dims['bbox'][2][0]}x{init_dims['bbox'][2][1]}"
        ))


def load_env(random_ng, num_obs, env_name, init_dims):
    import gym

    env = gym.make(env_name, **init_dims)

    return env


def classify_snr(snr_exp, div, count, num_env, lb=0):
    inc = 0
    if count[0] < num_env and (lb < snr_exp <= (div * 1 + lb)):
        count[0] += 1
        inc = 1
    elif count[1] < (num_env) and ((div * 1 + lb) < snr_exp <= (div * 2 + lb)):
        count[1] += 1
        inc = 1
    elif count[2] < num_env and ((div * 2 + lb) < snr_exp <= (div * 3 + lb)):
        count[2] += 1
        inc = 1
    elif count[3] < num_env and ((div * 3 + lb) < snr_exp <= (div * 4 + lb)):
        count[3] += 1
        inc = 1
    return count, inc


def set_vis_coord(point, coords):
    point.set_x(coords[0])
    point.set_y(coords[1])
    return point


def view_envs(path, max_obs, num_envs, env_name, init_dims, name, render=True):
    for jj in range(max_obs+1):
        print(f"----------------Num_obs {jj} ------------------")
        rng = np.random.default_rng(robust_seed)
        env = load_env(rng, jj, env_name=env_name, init_dims=init_dims)
        _ = env.reset()
        env_set = joblib.load(path + name)
        inter_count = 0
        repl = 0
        for kk in range(num_envs):
            env.refresh_environment(env_dict=env_set, id=kk)
            L = vis.Line_Segment(env.agents[0].detector, env.source)
            inter = False
            zz = 0
            # while not inter and zz < jj:
            #     if vis.boundary_distance(L, env.poly[zz]) < 0.001:
            #         inter = True
            #         inter_count += 1
            #     zz += 1

            if render and repl < 5:
                fig, ax1 = plt.subplots(1, figsize=(5, 5), tight_layout=True)
                ax1.scatter(env.src_coords[0], env.src_coords[1], c="red", marker="*")
                ax1.scatter(env.agents[0].det_coords[0], env.agents[0].det_coords[1], c="black")
                ax1.grid()
                ax1.set_xlim(0, env.search_area[1][0])
                ax1.set_ylim(0, env.search_area[1][0])
                for obs in env.obs_coord:
                    p_disp = Polygon(obs)
                    ax1.add_patch(p_disp)
                plt.show()
                repl += 1
        if jj == max_obs:
            jj += 1
        # print(f"Out of {num_envs} environments, {inter_count/num_envs:2.2%} have an obstruction between source and detector starting position.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_count", type=int, default=1000, help="Number of environments to generate")
    parser.add_argument("--max_obstacles", type=int, default=5, help="Generate environments with 0 to max_obstacles obstructions (inclusive)")
    parser.add_argument("--seed", type=int, default=500, help="Seed for randomization control")
    parser.add_argument("--dimension_max", type=list, default=[1500, 1500], help="Upper bound (cm) for x and y coordinates for environment")
    parser.add_argument("--test", type=str, default='0', help="Test env to run in simulation environment. Overwrites init_dims.")
    args = parser.parse_args()

    num_envs = args.env_count
    obs_list = [i for i in range(args.max_obstacles+1)]
    seed = args.seed
    snr_list = ["none", "low", "med", "high"]

    robust_seed = _int_list_from_bigint(hash_seed(seed))[0]
    rng = np.random.default_rng(robust_seed)

    print("Saving...")

    env_name = "gym_rad_search:RadSearchMulti-v1"

    if args.test in ['1', '2', '3', '4', 'ZERO']:
        num_obs = 0
        init_dims = {
            "bbox": [[0.0, 0.0], [args.dimension_max[0], 0.0], [args.dimension_max[0], args.dimension_max[1]], [0.0, args.dimension_max[1]]],
            "observation_area": [100.0, 200.0],
            "MIN_STARTING_DISTANCE": 500,
            "obstruction_count": num_obs,
            "np_random": rng,
            "TEST": args.test,
            "silent": True
        }
        # Obstructions are hard coded for test 1-4 and ZERO
        save_p = f"./test_evironments_TEST{args.test}/"
        load_p = f"./test_evironments_TEST{args.test}/"
        create_envs_snr(num_envs, init_dims, env_name, save_p, snr='none')
    else:
        save_p = f"./test_evironments_{args.dimension_max[0]}/"
        load_p = f"./test_evironments_{args.dimension_max[0]}/"
        for num_obs in obs_list:
            init_dims = {
                "bbox": [[0.0, 0.0], [args.dimension_max[0], 0.0], [args.dimension_max[0], args.dimension_max[1]], [0.0, args.dimension_max[1]]],
                "observation_area": [100.0, 200.0],
                "MIN_STARTING_DISTANCE": 500,
                "obstruction_count": num_obs,
                "np_random": rng,
                "TEST": 0
            }

            for snr in snr_list:
                create_envs_snr(num_envs, init_dims, env_name, save_p, snr=snr)

    print("Done")
