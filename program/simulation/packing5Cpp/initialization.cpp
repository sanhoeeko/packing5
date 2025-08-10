#include "pch.h"
#include "defs.h"
#include "potential.h"
#include "segdist.h"
#include <random>

/*
    return: 0 -> Success, -1 -> Fail
*/
int SegmentInitialization(void* xyt_ptr, int N, float A, float B, float r, int max_trial) {
    float r0 = 1 - r, gamma = 1 / r0;
    float rc = 2 * r0;
    float a = A - 1, b = B - 1;
    xyt* q = (xyt*)xyt_ptr;

    std::random_device rd;
    std::uniform_real_distribution<float> dist(0, 1);
    int cnt = 0;
    for (int i = 0; i < max_trial; i++) {
        if (cnt >= N)break;
        float
            R = sqrtf(dist(rd)),
            phi = 2 * pi * dist(rd),
            x = a * R * fcos(phi),
            y = b * R * fsin(phi),
            t = pi * dist(rd);
        bool accept = true;
        for (int j = 0; j < cnt; j++) {
            if (SegDist(r, x, y, t, q[j].x, q[j].y, q[j].t) < rc) {
                accept = false; break;
            }
        }
        if (accept) {
            q[cnt] = { x,y,t,0 }; 
            cnt++;
        }
    }
    cout << "Number of initialized particles:" << cnt << endl;
    if (cnt == N) {
        return 0;
    }
    else {
        cout << "Random Initialization failed!" << endl;
        return -1;
    }
}