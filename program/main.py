import simulation.boundary as boundary
from experiment import SimulationsCommonParticle
from simulation.potential import PowerFunc

if __name__ == '__main__':
    s = SimulationsCommonParticle(3, 0.05, PowerFunc(2.5), 4)
    s.appendSimulation(N=1024, phi0=0.5, Gamma0=1.0, compress_func_A=boundary.NoCompress(),
                       compress_func_B=boundary.RatioCompress(0.01), max_relaxation=1e4)
    # s.appendSimulation(N=1024, phi0=0.5, Gamma0=1.0, compress_func_A=boundary.NoCompress(),
    #                    compress_func_B=boundary.RatioCompress(0.01), max_relaxation=1e4)
    # s.appendSimulation(N=1024, phi0=0.5, Gamma0=1.0, compress_func_A=boundary.NoCompress(),
    #                    compress_func_B=boundary.RatioCompress(0.01), max_relaxation=1e4)
    # s.appendSimulation(N=1024, phi0=0.5, Gamma0=1.0, compress_func_A=boundary.NoCompress(),
    #                    compress_func_B=boundary.RatioCompress(0.01), max_relaxation=1e4)
    s.execute()
