import os
import sys

from h5tools import dataset as dset, h5tools as ht
from simulation import potential, boundary
from simulation.ensemble import CreateEnsemble


###################### functions start ################################################################

def start_ensemble(replica, N, n, d, phi0, Gamma0):
    """
    This function has type check. Parameters may be strings.
    """
    radial_func = potential.PowerFunc(2.5)
    compress_func_A = boundary.NoCompress()
    compress_func_B = boundary.RatioCompress(0.002)
    ensemble = CreateEnsemble(int(N), int(n), float(d), float(phi0), float(Gamma0),
                              compress_func_A, compress_func_B, radial_func)
    ensemble.setReplica(int(replica))
    ensemble.execute()


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
        keyword_args = {arg.split('=')[0]: arg.split('=')[1]
                        for arg in args if '=' in arg}
        call_function_by_name(func_name, *positional_args, **keyword_args)
    else:
        print("Please provide a function name.")
