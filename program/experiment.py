import itertools
import random
import subprocess
import sys
from collections.abc import Iterable

from h5tools.utils import current_time
from simulation import utils as ut


def run_command(func_name, *args, **kwargs):
    ut.setWorkingDirectory()
    cmd = [sys.executable, "shell.py", func_name]
    cmd.extend(list(map(str, args)))
    for k, v in kwargs.items():
        cmd.append(f"{k}={str(v)}")

    # create and open log
    log_filename = f"{str(random.randint(0, 1073741824))}.log"
    log_file = open(log_filename, "w")

    # start subprocess
    print("Run command:", " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
    return process, log_file


def call_start_ensemble(replica, *args, **kwargs):
    return run_command('start_ensemble', replica, *args, **kwargs)


def dict_cartesian_product(dicts: dict[str, list]) -> list[dict]:
    keys = dicts.keys()
    values = dicts.values()
    product_list = itertools.product(*values)
    result = [dict(zip(keys, combination)) for combination in product_list]
    return result


def ExperimentMain(replica: int, **kwargs):
    print('Simulation starts at', current_time())
    # preprocess of inputs
    for k, v in kwargs.items():
        if not isinstance(v, Iterable):
            kwargs[k] = [v]
    ensemble_kwargs = dict_cartesian_product(kwargs)

    # start subprocesses and create log files
    processes = []
    log_files = []
    for kwargs in ensemble_kwargs:
        process, log_file = call_start_ensemble(replica, **kwargs)
        processes.append(process)
        log_files.append(log_file)

    # wait for calculation
    for process, log_file in zip(processes, log_files):
        process.wait()
        log_file.close()

    print('Simulation ends at', current_time())
    run_command('pack_and_compress')
