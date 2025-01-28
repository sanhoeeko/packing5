#pragma once

#include<random>
#include<cstdlib>

template<int capacity>
struct MyRand {
    float data[capacity];

    MyRand() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
        for (int i = 0; i < capacity; i++) {
            float x = 114514;
            do {
                x = dis(gen);
            } while (abs(x) >= 3);       // such that |x| < 3 sigma
            data[i] = x;
        }
    }
    float get(unsigned int i) {
        return data[i % capacity];
    }
};

float fast_gaussian(int random_int);


struct xorshift32 {
    uint32_t state;

    xorshift32(uint32_t x);
    uint32_t operator()();
};