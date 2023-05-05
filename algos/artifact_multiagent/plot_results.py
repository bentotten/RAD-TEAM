"""
Create graph of training or evaluation cycle from saved results in models directory
"""
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import scipy.signal as signal  # type: ignore
import json
import os
import os.path as osp
import pandas as pd  # type: ignore

from typing import (
    Any,
    List,
    Tuple,
    Union,
    Literal,
    NewType,
    Optional,
    TypedDict,
    cast,
    get_args,
    Dict,
    Callable,
    overload,
    NamedTuple,
)

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units: Dict = dict()


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if "progress.txt" in files:
            exp_name = None
            # TODO do we need the config?
            # try:
            #     config_path = open(os.path.join(root, "general_s2/config.json"))
            #     config = json.load(config_path)
            #     if "exp_name" in config:
            #         exp_name = config["exp_name"]
            # except:
            #     print("No file named config.json")
            condition1 = condition or exp_name or "exp"
            condition2 = condition1 + "-" + str(exp_idx)
            exp_idx += 1
            # if condition1 not in units:
            #    units[condition1] = 0
            # unit = units[condition1]
            # units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, "progress.txt"))
            except:
                print("Could not read from %s" % os.path.join(root, "progress.txt"))
                continue
            # performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet' # Missing from dataset
            performance = "MeanEpRet"
            # exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns), "Condition1", condition1)
            exp_data.insert(len(exp_data.columns), "Condition2", condition2)
            try:
                exp_data.insert(len(exp_data.columns), "Performance", exp_data[performance])
            except:
                performance = "AverageEpRet"
                exp_data.insert(len(exp_data.columns), "Performance", exp_data[performance])
                
            datasets.append(exp_data)
    return datasets


def filt(sig, win):
    b = np.ones(win)
    a = np.array([win] + [0] * (win - 1))
    signal_filt = signal.filtfilt(b, a, sig)
    return signal_filt


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def multi_plot(data, smooth=None, x_axis="Epoch", save_f=False, file_name="."):
    ref_DF = pd.DataFrame()

    # lst = ['AverageEpRet','StdEpRet','DoneCount','EpLen','Entropy','kl_divergence', 'loss_predictor', 'loss_critic']  # 'AverageEpRet' Missing from dataset
    # list = [AgentID	Epoch	MeanVVals	StdVVals	MaxVVals	MinVVals	TotalEnvInteracts	loss_policy	loss_critic	loss_predictor	LocLoss	Entropy	kl_divergence	ClipFrac	OutOfBound	stop_iteration	MeanEpRet	StdEpRet	MaxEpRet	MinEpRet	DoneCount	EpLen	Time]
    lst = (
        data.columns
    )  # ['MeanEpRet','StdEpRet','DoneCount','EpLen','Entropy','kl_divergence', 'loss_predictor', 'loss_critic']
    exclude = ["Condition1", "Condition2", "AgentID", "Time", "Epoch"]
    include = ['MeanEpRet',	'StdEpRet',	'MaxEpRet',	'MinEpRet',	'DoneCount', 'EpLen']

    for lab in lst:
        if lab in include and lab not in exclude and np.any(lab == data.columns):
            ref_DF[lab] = data[data.columns[data.columns == lab][0]]
            if np.isnan(ref_DF[lab]).any():
                ref_DF[np.isnan(ref_DF[lab])] = 0
                print("NAN found!")
    iters = ref_DF.items()
    #if (len(lst) - len(exclude)) % 2 == 0:
    print(len(include))
    if len(include) % 2 == 0:
        div = 2
    else:
        div = 3
    # fig,axs = plt.subplots(len(lst)//div,len(lst)//(div+1),figsize=(12,8), constrained_layout=True)
    plt.rc("font", size=26)
    fig, axs = plt.subplots(
        div,
        len(include) // (div),
        figsize=(50, 16),
        constrained_layout=True,
    )
    x = data[data.columns[data.columns == x_axis][0]].to_numpy()
    for ax, case in zip(axs.flatten(), iters):
        if smooth < 0:
            d_filt = case[1]
        else:
            d_filt = filt(case[1], smooth)
        ax.set_ylabel("%s" % str(case[0]))
        ax.set_xlabel(x_axis)
        ax.plot(x, d_filt)

    if save_f:
        if file_name[-1] != "/":
            file_name += "/"
        fig.savefig(file_name + "results.png")
        fig.savefig(file_name + "results.eps")
    else:
        plt.show()


def plot(data, type_=None, smooth=True):
    x_ax = "Total Env Interacts"
    if type_ is "return":
        y_ax = "Total Return"
    elif type_ is "entropy":
        y_ax = "Entropy"
    elif type_ is "kl":
        y_ax = "kl_divergence"
    elif type_ is "len":
        y_ax = "Episode Length"
    elif type_ is "loss_v":
        y_ax = "Value Loss"
    elif type_ is "ex_var":
        y_ax = "Explained Var."
    else:
        y_ax = "Perf"

    if smooth:
        data = filt(data, 3)
    n = range(len(data))
    plt.figure()
    plt.plot(n, data)
    plt.ylabel(y_ax)
    plt.xlabel(x_ax)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where results are saved. Ex: ../models/train/gru_8_acts/bpf/model_dir",
        default="../../models/pre_train/gru_8_acts/bpf/loc32_hid32_pol32_val32_alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh_ep3000_steps4800_s1/",
    )
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument(
        "--smooth",
        type=int,
        default=-1,
        help="Moving average filter, if less than -1 than no averaging is applied",
    )
    args = parser.parse_args()
    dataset = get_datasets(args.data_dir)  # np.load(data_dir)
    multi_plot(
        dataset[0],
        smooth=args.smooth,
        x_axis="Epoch",
        save_f=args.save,
        file_name=args.data_dir,
    )
