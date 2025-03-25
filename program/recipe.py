from simulation.ensemble import SetRelaxationRecipe
from simulation.relaxation import Relaxation, Criterion
from simulation.stepsize import StepsizeHelper


def InitRecipe():
    SetRelaxationRecipe(
        Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=1, inertia=1, stepsize=1e-4, relaxation_steps=1e4,
                   state_pool_stride=1, auto_stepsize=StepsizeHelper.Nothing, enable_lbfgs=False, record_descent=True,
                   criterion=Criterion.EnergyFlat),
        Relaxation(noise_factor=0, momentum_beta=0, stochastic_p=1, inertia=1, stepsize=1, relaxation_steps=1e4,
                   state_pool_stride=1, auto_stepsize=StepsizeHelper.Best, enable_lbfgs=False, record_descent=True,
                   criterion=Criterion.MaxGradientAmp),
    )
