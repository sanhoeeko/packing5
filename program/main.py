from experiment import Experiment
from simulation import boundary

if __name__ == '__main__':
    ex = Experiment(
        replica=4,
        N=1000,
        n=3,
        d=0.025,
        Gamma0=1,
        phi0=0.5,
        compress_func_A=boundary.NoCompress(),
        compress_func_B=boundary.RatioCompress(0.01)
    )
    ex.execute()
