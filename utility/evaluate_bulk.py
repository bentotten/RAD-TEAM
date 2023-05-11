import os
import shutil
import subprocess
import sys
import multiprocessing
from functools import partial


PYTHON_PATH = sys.executable
SKIP_FILES = -1  # A way to skip directories that have already been processed

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


def get_paths(logdir, condition=None):
    """
    Recursively look through logdir for results files. Assumes that any file "progress.txt" is a valid hit. # noqa

    :param logdir: Root directory where data is contained.
    :param condition: Directory name to filter on. Ex: 'test1'
    """
    global exp_idx
    global units
    paths = []
    for root, _, files in os.walk(logdir):
        if "progress.txt" in files:
            last_dir = os.path.split(root)[-1]

            # If filtering on a condition, only read in data from said condition. # noqa
            if condition:
                if condition in last_dir:
                    paths.append(root)
            else:
                paths.append(root)
    return paths


def run(paths, components, groups, exclude, overwrite_flag, job_length, id):
    # Calcuate slice to process
    start_index = id * job_length
    stop_index = start_index + job_length

    # Begin evaluation
    for path in paths[start_index:stop_index]:
        # Get name
        name = os.path.split(path)[-1]
        (comp, test, excluded) = parse_exp_name(name=name, components=components, groups=groups, exclude=exclude)
        if not excluded and (not comp or not test):
            raise ValueError("Component or Test not in saved path name")
        if test and comp:
            test = test[-1]  # Get just test digit

            # Copy evaluate and saved_envs into test folder
            cwd = os.getcwd()
            if not os.path.isfile(path+'/evaluate.py') or overwrite_flag:
                shutil.copy(cwd+'/evaluate.py', path+'/evaluate.py')

            if not os.path.isfile(path+'/RADTEAM_core.py') or overwrite_flag:
                shutil.copy(cwd+'/RADTEAM_core.py', path+'/RADTEAM_core.py')

            if not os.path.isfile(path+'/core.py') or overwrite_flag:
                shutil.copy(cwd+'/core.py', path+'/core.py')

            if not os.path.isdir(path+'/saved_env/'):
                shutil.copytree(cwd+'/saved_env/', path+'/saved_env/')

            # Start evaluation
            print(f"### STARTING: {path}")
            og_dir = os.getcwd()
            os.chdir(path)

            launch = [PYTHON_PATH, "evaluate.py", "--rada2c", "--test", test]
            try:
                subprocess.run(launch)
            except Exception as e:
                print(e)

            os.chdir(og_dir)


def main(args):
    # Data directory formatting
    if args.data_dir == ".":
        args.data_dir = os.getcwd() + "/"
    args.data_dir = args.data_dir + '/' if args.data_dir[-1] != '/' else args.data_dir # noqa

    # Set up name parsing
    groups = ['test1', 'test2', 'test3', 'test4']

    # Groups to represent in each x tick group
    components = ['env', 'PPO', 'Optimizer', 'StatBuf', 'CNN']

    # Results to exclude from plotting
    exclude = ['RADMARL']

    # Get paths
    paths = get_paths(logdir=args.data_dir)

    overwrite_flag = True  # Flag to ensure overwrite of local copy of files happens every time

    # Set up multi-processing
    cpu_count = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpu_count)
    job_length = len(paths) / cpu_count

    thread_it = partial(run, paths, components, groups, exclude, overwrite_flag, job_length)

    # Start processes
    p.map(thread_it, range(0, cpu_count))

    # Plot test results
    subprocess.run(["python3", "plot_results.py"])

    print("Bulk evaluation complete.")


if __name__ == "__main__":
    ''' Get all paths to directories containing progress.txt and evaluate that directory ''' # noqa
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where results are saved. Ex: ../models/train/gru_8_acts/bpf/model_dir",  # noqa
        default=".",  # noqa
    )
    args = parser.parse_args()

    main(args)