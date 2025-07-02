import numpy as np
from pathos.multiprocessing import ProcessingPool

from analysis.database import Database, PickledEnsemble, PickledSimulation
from analysis.voronoi import DelaunayBase
from h5tools.h5analysis import add_array_to_hdf5
from h5tools.h5tools import write_metadata_to_hdf5


def process_simulation(tup: tuple[int, PickledSimulation]):
    index, simulation = tup
    print(simulation)
    results = []
    for i in range(len(simulation)):
        results.append(simulation.voronoi_at(i).delaunay())
    dic = {
        'metadata': simulation.metadata,
        'state_info': simulation.state_info,
        'voronoi': results
    }
    return index, dic


def ensemble_voronoi(out_file: str):
    def inner(ensemble: PickledEnsemble):
        indexed_tasks = list(enumerate(ensemble))
        indexed_tasks = indexed_tasks[:2]  # for test, or else it causes memory overflow

        with ProcessingPool() as pool:
            async_results = [pool.apipe(process_simulation, task) for task in indexed_tasks]
            unordered = [res.get() for res in async_results]
            sorted_results = [r[1] for r in sorted(unordered, key=lambda x: x[0])]

        metadata = sorted_results[0]['metadata']
        state_info = np.hstack([dic['state_info'] for dic in sorted_results])
        voronoi = sum([dic['voronoi'] for dic in sorted_results], [])
        indices, edges = zip(*list(map(extract_delaunay, voronoi)))

        # add metadata
        write_metadata_to_hdf5(out_file, metadata)
        # add data
        add_array_to_hdf5(out_file, 'state_info', state_info)
        add_array_to_hdf5(out_file, 'indices', indices)
        add_array_to_hdf5(out_file, 'edges', edges)

    return inner


def extract_delaunay(delaunay: DelaunayBase):
    return delaunay.indices, delaunay.edges


if __name__ == '__main__':
    db = Database('../data-20250419.h5')
    out_file = 'voronoi.h5'
    lst = db.apply(ensemble_voronoi(out_file))
    add_array_to_hdf5(out_file, 'summary_table', db._summary_table_array)  # add metadata
