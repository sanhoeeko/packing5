from simulation.ensemble import SetRelaxationRecipe
from simulation.relaxation import Relaxation, Criterion


# def InitRecipe():
#     SetRelaxationRecipe(
#         Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=1, stepsize=4e-5, relaxation_steps=1e5,
#                    state_pool_stride=1, auto_stepsize=False, record_energy=True, criterion=Criterion.EnergyFlat)
#     )


def InitRecipe():
    SetRelaxationRecipe(
        Relaxation(noise_factor=0.1, momentum_beta=0.9, stochastic_p=1, stepsize=1e-3, relaxation_steps=4e4,
                   state_pool_stride=1, auto_stepsize=False, record_energy=True, criterion=Criterion.NoCriterion),
        Relaxation(noise_factor=0.01, momentum_beta=0.1, stochastic_p=1, stepsize=4e-5, relaxation_steps=4e4,
                   state_pool_stride=20, auto_stepsize=False, record_energy=True, criterion=Criterion.MeanGradientAmp),
        Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=1, stepsize=1e-5, relaxation_steps=2e4,
                   state_pool_stride=1, auto_stepsize=False, record_energy=True, criterion=Criterion.MeanGradientAmp)
    )
