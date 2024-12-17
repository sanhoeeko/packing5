from simulation import boundary
from simulation.experiment import Experiment

if __name__ == '__main__':
    ex = Experiment(
        replica=3,
        N=1024,
        n=6,
        d=0.05,
        Gamma0=1,
        phi0=0.6,
        compress_func_A=boundary.NoCompress(),
        compress_func_B=boundary.RatioCompress(0.002)
    )
    ex.execute()
