from simulation import boundary
from simulation.experiment import Experiment

if __name__ == '__main__':
    ex = Experiment(
        replica=5,
        N=1024,
        n=[1+5,  1+10, 1+15, 1+20, 1+25, 1+30],
        d=0.02,
        Gamma0=1,
        phi0=0.6,
        compress_func_A=boundary.NoCompress(),
        compress_func_B=boundary.RatioCompress(0.002)
    )
    ex.execute()
