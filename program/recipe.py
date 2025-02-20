from simulation.ensemble import SetRelaxationRecipe
from simulation.relaxation import Relaxation, Criterion


def InitRecipe():
    SetRelaxationRecipe(
        Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=0, stepsize=1e-4, relaxation_steps=1e5,
                   state_pool_stride=1, record_stride=100, auto_stepsize=False, record_energy=True,
                   criterion=Criterion.EnergyFlat)
    )
