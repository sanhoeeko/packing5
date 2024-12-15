#pragma once

#include"defs.h"
#include<vector>
using namespace std;


vector<float> linspace(float start, float stop, int size);
vector<float> linspace_including_endpoint(float start, float stop, int size);

bool isnan(xyt& q);
bool isinf(xyt& q);
