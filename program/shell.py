import os
import sys

from analysis.analysis import calAllOrderParameters
from analysis.database import Database
from analysis.post_analysis import RawOrderDatabase, MergePostDatabase, MeanCIDatabase
from experiment import run_command
from h5tools import dataset as dset, h5tools as ht
from simulation.ensemble import StartEnsemble


###################### functions start ################################################################

def start_ensemble(replica, N, n, d, phi0, Gamma0):
    """
    This function has type check. Parameters may be strings.
    """
    StartEnsemble(replica, N, n, d, phi0, Gamma0)


def pack_and_compress():
    file_name = 'data.h5'
    sum_files = [file for file in os.listdir() if file.endswith('.h5')]
    ht.pack_h5_files(sum_files, file_name)
    for file in sum_files: os.remove(file)
    original_size, compressed_size = dset.compress_file(file_name)
    compress_rate_percent = int((1 - compressed_size / original_size) * 100)
    print(f"Successfully compressed. Compress rate: {compress_rate_percent}%.")


def auto_pack():
    dset.auto_pack()


def analyze(filename):
    full_file_name = 'full-' + filename
    mean_ci_file_name = 'analysis-' + filename
    calAllOrderParameters(Database(filename), 'phi', num_threads=4, averaged=False, out_file=full_file_name)
    RawOrderDatabase(full_file_name).mean_ci(mean_ci_file_name)


def batch_analyze(*filenames):
    # start subprocesses
    processes = []
    for filename in filenames:
        process, log_file = run_command('analyze', filename)
        processes.append(process)

    # wait for calculation
    for process in processes:
        process.wait()

    # merge results
    MergePostDatabase(RawOrderDatabase, 'merge-full.h5')(
        *list(filter(lambda x:x.startswith('full-'), os.listdir(os.getcwd())))
    )
    MergePostDatabase(MeanCIDatabase, 'merge-analysis.h5')(
        *list(filter(lambda x:x.startswith('analysis-'), os.listdir(os.getcwd())))
    )


###################### functions end ##################################################################

def call_function_by_name(func_name, *args, **kwargs):
    if func_name in globals():
        func = globals()[func_name]
        if callable(func):
            func(*args, **kwargs)
        else:
            print(f"{func_name} is not a callable function.")
    else:
        print(f"Function {func_name} not found.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        args = sys.argv[2:]
        positional_args = [arg for arg in args if '=' not in arg]
        keyword_args = {arg.split('=')[0]: arg.split('=')[1] for arg in args if '=' in arg}
        call_function_by_name(func_name, *positional_args, **keyword_args)
    else:
        print("Please provide a function name.")
