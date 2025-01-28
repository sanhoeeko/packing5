#include "pch.h"
#include "defs.h"
#include <cstring>

void AverageState(float temperature, void* p_states, void* p_energies, void* p_dst, int N, int n_samples)
{
    float* states = (float*)p_states;
    float* energies = (float*)p_energies;
    float* dst = (float*)p_dst;
    float* exps = new float[n_samples];

    float exps_tot = 0;
    for (int i = 0; i < n_samples; i++) {
        exps_tot += exps[i] = expf(-energies[i] / temperature);
    }
    FastClear(dst, 4 * N);
    for (int i = 0; i < n_samples; i++) {
        AddVector4(dst, states + i * (4 * N), dst, N, exps[i] / exps_tot);
    }
    delete[] exps;
}

float AverageStateZeroTemperature(void* p_states, void* p_energies, void* p_dst, int N, int n_samples)
{
    float* states = (float*)p_states;
    float* energies = (float*)p_energies;

    float current_e = energies[0];
    int current_idx = 0;
    for (int i = 1; i < n_samples; i++) {
        if (energies[i] <= current_e) {
            current_e = energies[i];
            current_idx = i;
        }
    }
    memcpy(p_dst, states + current_idx * (4 * N), 4 * N * sizeof(float));
    return current_e;
}