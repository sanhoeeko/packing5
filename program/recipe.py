from simulation.ensemble import SetRelaxationRecipe
from simulation.relaxation import Relaxation, Criterion


# def InitRecipe():
#     SetRelaxationRecipe(
#         Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=1, inertia=4, stepsize=4e-5, relaxation_steps=1e5,
#                    state_pool_stride=1, auto_stepsize=False, record_descent=True, criterion=Criterion.EnergyFlat)
#     )


def InitRecipe():
    SetRelaxationRecipe(
        # Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=1, inertia=1, stepsize=1e-4, relaxation_steps=1e4,
        #            state_pool_stride=1, auto_stepsize=False, record_descent=True, criterion=Criterion.EnergyFlat),
        Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=0.1, inertia=1, stepsize=1e-4, relaxation_steps=2e4,
                   state_pool_stride=100, auto_stepsize=False, record_descent=True, criterion=Criterion.EnergyFlat)
    )
