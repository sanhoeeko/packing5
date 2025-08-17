max_energy_curve_capacity = 1000000
terminal_grad_for_mean_gradient_amp = 1e-5
terminal_grad_for_max_gradient_amp = 1e-4
terminal_energy_slope = 1e-6
energy_counter_stride = 100
energy_counter_occurrence = 10
energy_eps = 1e-3
descent_curve_stride = 1
max_brown = 1e4
max_relaxation = 2e3
max_step_size = 1e-2
max_compress = 0.1
compress_rate = 6e-4  # for N=10000
uniform_compress_rate = 1e-3
max_compress_turns = 8000
if_cal_energy = True
if_affine_when_compress = False
if_hyperuniform_initialize = False
if_parallel_initialize = False
if_keep_boundary_aspect = False
enable_legal_check = False
S_local_background = 0.338
h_max = 1.2
phi_0 = 0.4
phi_c = 0.84
segdist_for_dense = 2.05
# segdist_for_sparse = 2.6  # for gamma=1.2
segdist_for_sparse = 3.4  # for gamma=1.8
