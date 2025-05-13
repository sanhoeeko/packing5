import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

import analysis.utils as ut
from analysis.database import Database
from analysis.kernel import ker
from analysis.orders import Delaunay


def angle57_dist(n_angles: int, state_dic: dict, delaunay: Delaunay) -> (np.ndarray, np.ndarray):
    angles = np.linspace(0, np.pi / 2, n_angles, endpoint=False)
    counts = ut.CArray(np.zeros((n_angles,), dtype=np.int32))
    xyt_c = ut.CArray(state_dic['xyt'])
    z = ut.CArray(delaunay.z_number())
    ker.dll.Angle57Dist(*delaunay.params, n_angles, xyt_c.ptr, z.ptr, counts.ptr)
    return angles, counts.data


def process_task(task):
    """
    Process one task which corresponds to a chunk (given by chunk_index) from a specific database file and gamma.
    Opens the database file within the worker so that h5py objects remain local.
    """
    db_filename, chunk_index, gamma = task
    db = Database(db_filename)
    e = db.find(gamma=gamma)[0]

    simu = e[chunk_index]  # simu: PickledSimulation
    local_counts = np.zeros((90,), dtype=np.int32)
    lower_index, upper_index = ut.indexInterval(simu.state_info['phi'], simu.metadata['gamma'], 0.85, 1.2)

    for j in range(lower_index, upper_index):
        angles, counts = angle57_dist(90, simu[j], simu.voronoi_at(j).delaunay())
        local_counts += counts
    return local_counts


if __name__ == '__main__':
    # List all database filenames.
    db_files = [
        '../data-20250420.h5',
        '../data-20250420-2.h5'
    ]

    # Define gamma values from 1.1 to 3.0 in steps of 0.1.
    gammas = np.arange(1.1, 3.1, 0.1)

    # fot test
    gammas = [1.6]

    # To store the result for each gamma; each entry is a vector of length 90.
    results_by_gamma = []

    # Loop serially over gamma values.
    for gamma in gammas:
        # Create one task per (database, chunk) for this gamma.
        tasks = [(db_filename, chunk_index, gamma)
                 for db_filename in db_files
                 for chunk_index in range(5)]

        # Create a pool with a worker for each task.
        with multiprocessing.Pool(processes=len(tasks)) as pool:
            # Process all tasks in parallel with a single pool.map call.
            results = pool.map(process_task, tasks)

        # Sum up all tasks for the current gamma to form a single distribution.
        total_counts = np.sum(results, axis=0)
        results_by_gamma.append(total_counts)

    # Stack results_by_gamma into a matrix.
    # Shape: (number of gamma values, n_angles=90)
    final_matrix = np.vstack(results_by_gamma)

    # Save final matrix
    np.save('defect_angle_dist.npy', final_matrix)

    # for test
    plt.plot(final_matrix.T)
    plt.show()
