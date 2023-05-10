import matplotlib.pyplot as plt  # type: ignore
import os
from typing import Dict
import json

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units: Dict = dict()


def get_datasets(logdir, condition=None):
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

                    with open(os.path.join(root, "results.json"), 'r') as f:
                        exp_data = json.load(f)
                    f.close()

                    exp_data['experiment'] = last_dir

                    datasets.append(exp_data)
    return datasets


def parse_data(data, item_names, graph_names):
    ''' Parse data into plt compatible datasets '''

    graphs = {name: None for name in graph_names}

    for item in data:
        pass


def plot_data(data):
    pass


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
    parser.add_argument("--condition", type=str, default='test1', help='Condition to filter on when reading in data')

    args = parser.parse_args()

    if args.data_dir == '.':
        args.data_dir = os.getcwd() + '/'

    headings = []

    raw_data = get_datasets(logdir=args.data_dir, conditions=args.condition)
    
    data = parse_data(data=raw_data)
    
    
    plot(data)