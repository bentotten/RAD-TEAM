import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import os
from typing import Dict
import json
import pandas as pd
from typing import NewType, List
import numpy.typing as npt
import matplotlib.ticker as mticker


# Global vars for tracking and labeling data at load time.
exp_idx = 0
units: Dict = dict()
Color = NewType("Color", npt.NDArray[np.float64])
Colorcode = NewType("Colorcode", List[int])

COLOR_FACTOR = 0.75  # How much to lighten previous color by
COLORS = [
    # Colorcode([148, 0, 211]), # Violet (Removed due to being too similar to indigo)
    Colorcode([255, 105, 180]),  # Pink
    Colorcode([75, 0, 130]),  # Indigo
    Colorcode([0, 0, 255]),  # Blue
    Colorcode([0, 255, 0]),  # Green
    Colorcode([255, 127, 0]),  # Orange
]

# Helper Functions
def create_color(id: int) -> Color:
    """Pick initial Colorcode based on id number, then offset it"""
    specific_color: Colorcode = COLORS[id % (len(COLORS))]  #
    if id > (len(COLORS) - 1):
        offset: int = (id * 22) % 255  # Create large offset for that base color, bounded by 255
        specific_color[id % 3] = (255 + specific_color[id % 3] - offset) % 255  # Perform the offset
    return Color(np.array(specific_color) / 255)


def lighten_color(color: Color, factor: float) -> Color:
    """increase tint of a color"""
    scaled_color = color * 255  # return to original scale
    return Color(np.array(list(map(lambda c: (c + (255 - c) * factor) / 255, scaled_color))))


def parse_exp_name(name, components, groups, exclude):
    first_split = name.split(' ')
    parts = []

    for name in first_split:
        parts += name.split('_')
    result = None
    category = None
    exclude_flag = False
    excluded = []
    for part in parts:
        if part in exclude:
            exclude_flag = True
            excluded += part
        elif part in components:
            assert result is None, "Multiple results match"
            result = part
        elif part in groups:
            assert category is None, "Multiple categories match"
            category = part

    return (result, category, None) if not exclude_flag else (None, None, excluded)


def get_data(logdir, condition=None):
    """
    Recursively look through logdir for results files. Assumes that any file "results_raw.json" is a valid hit.

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
            else:
                with open(os.path.join(root, "results.json"), "r") as f:
                    exp_data = json.load(f)
                f.close()

                exp_data["experiment"] = last_dir

                datasets.append(exp_data)
    return datasets


def parse_data(data, components, groups, exclude, performance_markers):
    """Parse data into plt compatible datasets"""

    # Goal: have 3 graphs (lists), each containing each component (list) that have a tests (dict of stats):

    # TODO use sorting method and statically generated arrays instead of two loops and dicts
    # Eventual graphs to make - because run may be out of order, making a dict first
    accuracy_datasets = [[None for _ in range(len(groups))] for _ in range(len(components))]
    speed_datasets = [[None for _ in range(len(groups))] for _ in range(len(components))]
    score_datasets = [[None for _ in range(len(groups))] for _ in range(len(components))]

    # accuracy_sorted = {component: {test: None for test in groups} for component in components}
    # speed_sorted = {component: {test: None for test in groups} for component in components}
    # score_sorted = {component: {test: None for test in groups} for component in components}

    accuracy_sorted = {component: {} for component in components}
    speed_sorted = {component: {} for component in components}
    score_sorted = {component: {} for component in components}
    
    # Pulled from all different files
    for run in data:
        # Parse which component this is
        (comp, test, excluded) = parse_exp_name(name=run['experiment'], components=components, groups=groups, exclude=exclude)

        # Add component results to graph dicts (if not an excluded result)
        if comp and test:
            # Only one test per save, thus [0]
            accuracy_sorted[comp][test] = run['accuracy'][0] if isinstance(run['accuracy'], list) else run['accuracy']
            speed_sorted[comp][test] = run['speed'][0] if isinstance(run['speed'], list) else run['speed']
            score_sorted[comp][test] = run['score'][0] if isinstance(run['score'], list) else run['score']

    # Rearrange in specified order
    for ci, component in enumerate(components):
        for ti, test in enumerate(groups):
            accuracy_datasets[ci][ti] = accuracy_sorted[component][test]
            speed_datasets[ci][ti] = speed_sorted[component][test]
            score_datasets[ci][ti] = score_sorted[component][test]

    return accuracy_datasets, speed_datasets, score_datasets


def mock_data():
    low_whisker = 6.8
    q1 = 6.9
    median = 7.0
    q3 = 8.0
    high_whisker = 8.59

    # Set up stats way
    athens = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
    ]
    beijing = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
    ]
    london = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
    ]
    rio = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
    ]

    return [athens, beijing, london, rio]


def mock_data(small):
    low_whisker = 6.8
    q1 = 6.9
    median = 7.0
    q3 = 8.0
    high_whisker = 8.59

    # Set up stats way
    athens = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},        
    ]
    beijing = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker}
    ]
    london = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker}
    ]
    rio = [
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker},
        {'med': median, 'q1': q1, 'q3': q3, 'whislo': low_whisker, 'whishi': high_whisker}
    ]

    return [athens]


def plot(graphname, datasets, groups, tests, y_label, path=None):
    # Datasets correspond to components. Groups correspond to tests
    
    # Make figures A6 in size
    A = 6
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    # Use Latex
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Define which colours you want to use
    colors = []
    for id in range(len(groups)):
        colors.append(create_color(id))

    # Set x-positions for boxes
    x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1) if len(datasets) > 1 else np.array(1)
    x_pos = (x_pos_range * 0.5) + 0.75 if x_pos_range.size > 1 else x_pos_range

    _, ax = plt.subplots()

    # Set labels
    ax.set_xticklabels(tests)

    # Plot each component
    for i, data_to_plot in enumerate(datasets):
        if len(data_to_plot[0]) != 0:
            # positions = [x_pos[i] + j * 1 for j in range(len(data.T))]
            if x_pos_range.size > 1:
                positions = [x_pos[i] + j * 1 for j in range(len(data_to_plot))]
            else:
                positions = [x_pos + j * 1 for j in range(len(data_to_plot))]

            # TODO do this in data parser
            # stats = list()
            # for test in data:
            #     stats.append({'med': data[test]['med'], 'q1':  data[test]['q1'], 'q3':  data[test]['q3'], 'whislo':  data[test]['whislo'], 'whishi': data[test]['whishi']})

            bp = ax.bxp(
                # stats,
                data_to_plot,
                showfliers=False,
                positions=positions,
                patch_artist=True,
                widths=0.6 / len(datasets),
                )

            # Fill the boxes with colours (requires patch_artist=True)
            k = i % len(colors)
            for box in bp['boxes']:
                box.set(facecolor=colors[k])

            # Make the median lines more visible
            plt.setp(bp['medians'], color='red')

    # Titles
    plt.ylabel(y_label)

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
        k = i % len(colors)
        legend_elements.append(Patch(facecolor=colors[k], label=groups[j]))
    plt.legend(handles=legend_elements, fontsize=8)

    if path:
        plt.savefig(str(path) + f"/{graphname}_result.png", format="png")
        #plt.savefig(str(path) + "/evaluation_result.eps", format="eps")
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
    args = parser.parse_args()

    if args.data_dir == ".":
        args.data_dir = os.getcwd() + "/"
    args.data_dir = args.data_dir + '/' if args.data_dir[-1] != '/' else args.data_dir

    # Groups to sort by
    tests = ['zero', 'full'] # 'test3', 'test4']
    # agent_counts = ['1agent', '2agent', '4agent']
    # modes = ['collab', 'coop', 'control']

    groups = tests

    # Groups to represent in each x tick group
    components = ['control']
    # components = ['1agent', '2agent', '4agent']

    # Results to exclude from plotting
    exclude = ['agent']
    
    # Conditions for file read-in
    condition = '10k'

    performance_markers = {
        'accuracy': "Objective Completion %",
        'score': "Episode Cumulative Return [raw]",
        'speed': "Successful Episode Length [samples]"
        }

    data = get_data(logdir=args.data_dir, condition=condition)

    accuracy_datasets, speed_datasets, score_datasets = parse_data(data=data, components=components, groups=groups, exclude=exclude, performance_markers=performance_markers)

    print(accuracy_datasets)
    print(mock_data(True))
    # accuracy_datasets, speed_datasets, score_datasets = mock_data(True), mock_data(True), mock_data(True)

    for graphname, graph in zip([performance_markers['accuracy'], performance_markers['speed'], performance_markers['score']], [accuracy_datasets, speed_datasets, score_datasets]):
        # try:
        plot(graphname=graphname, datasets=graph, groups=components, tests=groups, y_label=graphname, path=os.getcwd())
        # except Exception as e:
        #     print(e)

    print("Done with plot")
