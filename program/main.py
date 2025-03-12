import numpy as np

from experiment import ExperimentMain
from recipe import InitRecipe

InitRecipe()

if __name__ == '__main__':
    ex = ExperimentMain(
        replica=5,
        N=1024,
        n=np.arange(5, 81 + 4, 4),
        d=0.05,
        Gamma0=1,
        phi0=0.4
    )
