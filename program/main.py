from experiment import ExperimentMain

if __name__ == '__main__':
    ex = ExperimentMain(
        replica=5,
        N=1024,
        n=[1 + 5, 1 + 10, 1 + 15, 1 + 20, 1 + 25, 1 + 30],
        d=0.02,
        Gamma0=1,
        phi0=0.7
    )
