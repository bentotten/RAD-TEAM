''' Quickly-generated bulk evaluation tool. '''
import os
import shutil
import subprocess
import sys


PYTHON_PATH = sys.executable


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

    print(args)

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

    # Begin evaluation
    for path in paths:
        # Get name
        name = os.path.split(path)[-1]
        (comp, test, excluded) = parse_exp_name(name=name, components=components, groups=groups, exclude=exclude)
        if not excluded and (not comp or not test):
            raise ValueError("Component or Test not in saved path name")
        test = test[-1]  # Get just test digit

        # Copy evaluate and saved_envs into test folder
        cwd = os.getcwd()
        if not os.path.isfile(path+'/evaluate.py'):
            shutil.copy(cwd+'/evaluate.py', path+'/evaluate.py')

        if not os.path.isfile(path+'/RADTEAM_core.py'):
            shutil.copy(cwd+'/RADTEAM_core.py', path+'/RADTEAM_core.py')

        if not os.path.isfile(path+'/plot_results.py'):
            shutil.copy(cwd+'/plot_results.py', path+'/plot_results.py')

        if not os.path.isdir(path+'/saved_env/'):
            shutil.copytree(cwd+'/saved_env/', path+'/saved_env/')

        # Start evaluation
        og_dir = os.getcwd()
        os.chdir(path)

        launch = [PYTHON_PATH, "evaluate.py", "--rada2c", "--test", test]

        subprocess.run(launch)
        subprocess.run(["python3", "plot_results.py"])
        os.chdir(og_dir)

        print("Bulk evaluation complete.")
