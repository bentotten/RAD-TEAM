import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import os
from typing import Dict
import json
import pandas as pd

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units: Dict = dict()


def get_data(logdir, condition=None):
    """
    Recursively look through logdir for results files. Assumes that any file "results_raw.txt" is a valid hit.

    :param logdir: Root directory where data is contained.
    :param condition: Directory name to filter on. Ex: 'test1'
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if "results.json" in files:
            last_dir = os.path.split(root)[-1]

            # If filtering on a condition, only read in data from said condition.
            if condition:
                if condition in last_dir:
                    exp_idx += 1

                    with open(os.path.join(root, "results.json"), "r") as f:
                        exp_data = json.load(f)
                    f.close()

                    exp_data["experiment"] = last_dir

                    datasets.append(exp_data)
    return datasets


def parse_data(data, components, x_ticks, metrics, performance_markers):
    """Parse data into plt compatible datasets"""

    data = [{
        'experiment': 'test1_1agent',
        "accuracy": {
            "low_whisker": 100.0,
            "q1": 100.0,
            "median": 100.0,
            "q3": 100.0,
            "high_whisker": 100.0,
            "std": 0.0
        },
        "super": {},
        "score": {
            "low_whisker": 0.55,
            "q1": 1.0,
            "median": 1.0,
            "q3": 1.0,
            "high_whisker": 1.0,
            "std": 0.151
        },
        "speed": {
            "low_whisker": 21.0,
            "q1": 22.0,
            "median": 22.0,
            "q3": 23.0,
            "high_whisker": 27.0,
            "std": 1.836
        }
    }]
    
    low_whisker = 6.8
    q1 = 6.9
    median = 7.0
    q3 = 8.0
    high_whisker = 8.59    

    athens = pd.DataFrame({
        'Test 1':  {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        'Test 2': {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        # 'Test 3': [],
        # 'Test 4': [],
    })
    beijing = pd.DataFrame({
        'Test 1': {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        'Test 2':{'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        # 'Test 3': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73],
        # 'Test 4': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73],
    })
    london = pd.DataFrame({
        'Test 1':  {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        'Test 2':{'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        # 'Test 3': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73],
        # 'Test 4': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73],
    })
    rio = pd.DataFrame({
        'Test 1':  {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        'Test 2':{'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        # 'Test 3': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73],
        # 'Test 4': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73],
    })


    # Results of the long jump finals at four Olympic games
    # athens = pd.DataFrame({
    #     'Men': [8.59, 8.47, 8.32, 8.31, 8.25, 8.24, 8.23, 8.21],
    #     'Women': [7.07, 7.05, 7.05, 6.96, 6.85, 6.83, 6.80, 6.73]
    # })
    # beijing = pd.DataFrame({
    #     'Men': [8.34, 8.24, 8.20, 8.19, 8.19, 8.16, 8.07, 8.00],
    #     'Women': [7.04, 7.03, 6.91, 6.79, 6.76, 6.70, 6.64, 6.58]
    # })
    # london = pd.DataFrame({
    #     'Men': [8.31, 8.16, 8.12, 8.11, 8.10, 8.07, 8.01, 7.93],
    #     'Women': [7.12, 7.07, 6.89, 6.88, 6.77, 6.76, 6.72, 6.67]
    # })
    # rio = pd.DataFrame({
    #     'Men': [8.38, 8.37, 8.29, 8.25, 8.17, 8.10, 8.06, 8.05],
    #     'Women': [7.17, 7.15, 7.08, 6.95, 6.81, 6.79, 6.74, 6.69]
    # })

    return [athens, beijing, london, rio]


def plot(datasets, components, x_ticks, metrics, performance_markers, path=None):

    # Fill in data with saved values
    low_whisker = 6.8
    q1 = 6.9
    median = 7.0
    q3 = 8.0
    high_whisker = 8.59

    stats = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker}
    ]    

    # Make figures A6 in size
    A = 6
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    # Use Latex
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Define which colours you want to use
    colours = ['blue', 'red', 'green', 'brown']
    # Define the groups
    groups = ['Athens 2004', 'Beijing 2008', 'London 2012', 'Rio 2016']

    # Set x-positions for boxes
    x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1)
    x_pos = (x_pos_range * 0.5) + 0.75
    
    _, ax = plt.subplots()
    
    # Plot
    for i, data in enumerate(datasets):
        positions = [x_pos[i] + j * 1 for j in range(len(data.T))]
        # bp = plt.boxplot(
        #     np.array(data),
        #     sym='',
        #     whis=[0, 100],
        #     widths=0.6 / len(datasets),
        #     labels=list(datasets[0]),
        #     patch_artist=True,
        #     positions=positions
        # )
        bp = ax.bxp(
            stats,
            showfliers=False,
            positions=positions,
            patch_artist=True,
            widths=0.6 / len(datasets),
            )

        # Fill the boxes with colours (requires patch_artist=True)
        k = i % len(colours)
        for box in bp['boxes']:
            box.set(facecolor=colours[k])
        # Make the median lines more visible
        plt.setp(bp['medians'], color='black')

        # Get the samples' medians
        medians = [bp['medians'][j].get_ydata()[0] for j in range(len(data.T))]
        medians = [str(round(s, 2)) for s in medians]

    # Titles
    plt.title('Long Jump Finals at the Last Four Olympic Games')
    plt.ylabel('Distance [m]')
    
    # Axis ticks and labels
    plt.xticks(np.arange(len(list(datasets[0]))) + 1)
    plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(
        np.array(range(len(list(datasets[0])) + 1)) + 0.5)
    )
    plt.gca().tick_params(axis='x', which='minor', length=4)
    plt.gca().tick_params(axis='x', which='major', length=0)
    
    # Change the limits of the x-axis
    plt.xlim([0.5, len(list(datasets[0])) + 0.5])
    
    # Legend
    legend_elements = []
    for i in range(len(datasets)):
        j = i % len(groups)
        k = i % len(colours)
        legend_elements.append(Patch(facecolor=colours[k], label=groups[j]))
    plt.legend(handles=legend_elements, fontsize=8)

    if path:
        plt.savefig(str(path) + "/evaluation_result.png", format="png")
        plt.savefig(str(path) + "/evaluation_result.eps", format="eps")
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
    parser.add_argument("--condition", type=str, default="test1", help="Condition to filter on when reading in data")

    args = parser.parse_args()

    if args.data_dir == ".":
        args.data_dir = os.getcwd() + "/"

    x_ticks = ['test1', 'test2', 'test3', 'test4']

    components = ['env', 'PPO', 'Optimizer', 'StatBuf', 'CNN']
    metrics = ['low_whisker', 'q1', 'median', 'q3', 'high_whisker', 'std']
    performance_markers = {
        'accuracy': "Objective Completion %",
        'score': "Episode Cumulative Return [raw]",
        'speed': "Successful Episode Length [samples]"
        }

    # data = get_data(logdir=args.data_dir, conditions=args.condition)
    data = [{
        'experiment': 'test1_1agent',
        "accuracy": {
            "low_whisker": 100.0,
            "q1": 100.0,
            "median": 100.0,
            "q3": 100.0,
            "high_whisker": 100.0,
            "std": 0.0
        },
        "super": {},
        "score": {
            "low_whisker": 0.55,
            "q1": 1.0,
            "median": 1.0,
            "q3": 1.0,
            "high_whisker": 1.0,
            "std": 0.151
        },
        "speed": {
            "low_whisker": 21.0,
            "q1": 22.0,
            "median": 22.0,
            "q3": 23.0,
            "high_whisker": 27.0,
            "std": 1.836
        }
    }]

    datasets = parse_data(data=data, components=components, x_ticks=x_ticks, metrics=metrics, performance_markers=performance_markers)

    plot(datasets=datasets, components=components, x_ticks=x_ticks, metrics=metrics, performance_markers=performance_markers, path=os.getcwd())

    print("Done")
