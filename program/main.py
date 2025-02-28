from experiment import ExperimentMain
from recipe import InitRecipe

InitRecipe()

if __name__ == '__main__':
    ex = ExperimentMain(
        replica=5,
        N=1024,
        n=[1 + 5, 1 + 10, 1 + 15, 1 + 20, 1 + 25, 1 + 30,
           1 + 35, 1 + 40, 1 + 45, 1 + 50, 1 + 55, 1 + 60,
           1 + 65, 1 + 70, 1 + 75, 1 + 80, 1 + 85, 1 + 90, 1 + 95, 1 + 100],
        d=0.02,
        Gamma0=1,
        phi0=0.4
    )
