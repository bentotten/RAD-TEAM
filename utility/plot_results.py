"""
Create graph of training cycle from saved results in models directory named "progress.txt"
"""
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import scipy.signal as signal  # type: ignore
import os
import pandas as pd  # type: ignore

from typing import (
    Dict,
)

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units: Dict = dict()


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


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
            condition1 = condition or exp_name or "exp"
            condition2 = condition1 + "-" + str(exp_idx)
            exp_idx += 1

            try:
                exp_data = pd.read_table(os.path.join(root, "progress.txt"))
            except:  # noqa
                print("Could not read from %s" % os.path.join(root, "progress.txt"))
                continue
            # performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet' # Missing from dataset
            performance = "MeanEpRet"
            # exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns), "Condition1", condition1)
            exp_data.insert(len(exp_data.columns), "Condition2", condition2)
            try:
                exp_data.insert(len(exp_data.columns), "Performance", exp_data[performance])
            except:  # noqa
                performance = "AverageEpRet"
                exp_data.insert(len(exp_data.columns), "Performance", exp_data[performance])

            datasets.append(exp_data)
    return datasets


def filt(sig, win):
    b = np.ones(win)
    a = np.array([win] + [0] * (win - 1))
    signal_filt = signal.filtfilt(b, a, sig)
    return signal_filt


def multi_plot(data, smooth=None, x_axis="Epoch", save_f=False, file_name="."):
    ref_DF = pd.DataFrame()

    lst = data.columns  # ['MeanEpRet','StdEpRet','DoneCount','EpLen','Entropy','kl_divergence', 'loss_predictor', 'loss_critic']
    exclude = ["Condition1", "Condition2", "AgentID", "Time", "Epoch", 'MaxEpRet', 'MinEpRet', 'AverageVVals',	'StdVVals',
               'MaxVVals',	'MinVVals',	'TotalEnvInteracts', 'LossPi',	'LossV',	'LossModel', 'LocLoss',	'Entropy',	'KL',
               'ClipFrac',	'OutOfBound',	'StopIter',	'Time']
    include = ["AverageEpRet", "StdEpRet", "DoneCount", "EpLen"]

    for lab in lst:
        if lab in include and lab not in exclude and np.any(lab == data.columns):
            ref_DF[lab] = data[data.columns[data.columns == lab][0]]
            if np.isnan(ref_DF[lab]).any():
                ref_DF[np.isnan(ref_DF[lab])] = 0
                print("NAN found!")
    iters = ref_DF.items()
    # if (len(lst) - len(exclude)) % 2 == 0:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where results are saved. Ex: ../models/train/gru_8_acts/bpf/model_dir",
        default=".",  # noqa
    )
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument(
        "--smooth",
        type=int,
        default=-1,
        help="Moving average filter, if less than -1 than no averaging is applied",
    )
    args = parser.parse_args()

    if args.data_dir == '.':
        args.data_dir = os.getcwd() + '/'

    dataset = get_datasets(args.data_dir)  # np.load(data_dir)
    multi_plot(
        dataset[0],
        smooth=args.smooth,
        x_axis="Epoch",
        save_f=args.save,
        file_name=args.data_dir,
    )
