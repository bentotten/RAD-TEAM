import gym
import numpy as np
import joblib
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
            env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity, env.obs_coord)
        else:
            env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity)
            print(f"Source coord: {env.src_coords}, Det coord: {env.det_coords}, Intensity: {env.intensity},{env.bkg_intensity}")

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
        det = np.linalg.norm(env.src_coords - env.det_coords)
        meas = env.intensity / (det**2) + env.bkg_intensity
        if snr == "none":
            if init_dims["obstruct"] > 0 or init_dims["obstruct"] == -1:
                env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity, env.obs_coord)
                ii += 1
            else:
                env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity)
                ii += 1
        else:
            snr_exp = meas / env.bkg_intensity
            if snr_range[snr][0] < snr_exp <= snr_range[snr][1]:
                if snr == "med" or snr == "low" or snr == "high":
                    counts, inc_flag = classify_snr(np.round(snr_exp, 3), div, counts, num_envs_split, lb=snr_range[snr][0])
                    if init_dims["obstruct"] > 0 or init_dims["obstruct"] == -1:
                        if inc_flag:
                            env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity, env.obs_coord)
                            ii += 1
                            if (ii % 100) == 0:
                                print(f"Obs SNR: {np.round(snr_exp,3)} -> {counts}")
                    else:
                        if inc_flag:
                            env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity)
                            ii += 1
                            if (ii % 100) == 0:
                                print(f"SNR: {np.round(snr_exp,3)} -> {counts}")
                else:
                    env_dict["env_" + str(ii)] = (env.src_coords, env.det_coords, env.intensity, env.bkg_intensity)
                    # print(f'Source coord: {env.src_coords}, Det coord: {env.det_coords}, Intensity: {env.intensity},{env.bkg_intensity}')
                    ii += 1
                    print(f"SNR: {np.round(snr_exp,3)}")

    joblib.dump(env_dict, osp.join(save_path, 'test_env_obs'+str(init_dims['obstruct'])+'_'+snr+'_v4'))


def load_env(random_ng, num_obs):
    import gym

    init_dims = {
        "bbox": [[0.0, 0.0], [2700.0, 0.0], [2700.0, 2700.0], [0.0, 2700.0]],
        "area_obs": [200.0, 500.0],
        "obstruct": num_obs,
        "seed": random_ng,
    }
    env_name = "gym_radloc:RadLoc-v0"
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


def view_envs(path, max_obs, num_envs, render=True):
    for jj in range(1, max_obs):
        print(f"----------------Num_obs {jj} ------------------")
        rng = np.random.default_rng(robust_seed)
        env = load_env(rng, jj)
        _ = env.reset()
        env_set = joblib.load(path + str(jj))
        inter_count = 0
        repl = 0
        for kk in range(num_envs):
            env.refresh_environment(env_dict=env_set, id=kk, num_obs=jj)
            L = vis.Line_Segment(env.detector, env.source)
            inter = False
            zz = 0
            while not inter and zz < jj:
                if vis.boundary_distance(L, env.poly[zz]) < 0.001:
                    inter = True
                    inter_count += 1
                zz += 1

            if render and repl < 5:
                fig, ax1 = plt.subplots(1, figsize=(5, 5), tight_layout=True)
                ax1.scatter(env.src_coords[0], env.src_coords[1], c="red", marker="*")
                ax1.scatter(env.det_coords[0], env.det_coords[1], c="black")
                ax1.grid()
                ax1.set_xlim(0, env.search_area[1][0])
                ax1.set_ylim(0, env.search_area[1][0])
                for coord in env.obs_coord:
                    p_disp = Polygon(coord[0])
                    ax1.add_patch(p_disp)
                plt.show()
                repl += 1
        print(f"Out of {num_envs} {inter_count/num_envs:2.2%} have an obstruction between source and detector starting position.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_count", type=int, default=1000, help="Number of environments to generate")
    parser.add_argument("--max_obstacles", type=int, default=5, help="Generate environments with 0 to max_obstacles obstructions")
    parser.add_argument("--seed", type=int, default=500, help="Seed for randomization control")
    args = parser.parse_args()

    num_envs = args.env_count
    obs_list = [i for i in range(args.max_obstacles)]
    seed = args.seed
    snr_list = ["none", "low", "med", "high"]

    robust_seed = _int_list_from_bigint(hash_seed(seed))[0]
    rng = np.random.default_rng(robust_seed)

    print("Saving...")
    for num_obs in obs_list:
        init_dims = {
            "bbox": [[0.0, 0.0], [2700.0, 0.0], [2700.0, 2700.0], [0.0, 2700.0]],
            "area_obs": [200.0, 500.0],
            "obstruct": num_obs,
            "seed": rng,
        }

        env_name = "gym_rad_search:RadSearchMulti-v1"
        save_p = "./test_evironments/"
        load_p = "./test_evironments/"

        for snr in snr_list:
            create_envs_snr(num_envs, init_dims, env_name, save_p, snr=snr)

        view_envs(load_p, num_obs, num_envs)

    print("Done")
