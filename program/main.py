# if __name__ == '__main__':
#     from experiment import Experiment
#     from simulation import boundary
#     ex = Experiment(
#         replica=4,
#         N=1024,
#         n=3,
#         d=0.025,
#         Gamma0=1,
#         phi0=0.5,
#         compress_func_A=boundary.NoCompress(),
#         compress_func_B=boundary.RatioCompress(0.01)
#     )
#     ex.execute()


if __name__ == '__main__':
    from analysis.database import Database
    from art.viewer import RenderSetup, InteractiveViewer

    db = Database('data.h5')
    InteractiveViewer(db.id('gbwi'), RenderSetup('z_number', False, 'voronoi', True)).show()
