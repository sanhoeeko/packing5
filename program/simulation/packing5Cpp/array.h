#pragma once

#include"defs.h"
#include<vector>
using namespace std;


vector<float> linspace(float start, float stop, int size);
vector<float> linspace_including_endpoint(float start, float stop, int size);

bool isnan(xyt& q);
bool isinf(xyt& q);


/*
    UnaryPredicate: type of function that returns a boolean.
    [pred]: the condition for items filtered OUT.
*/
template <typename T, typename UnaryPredicate>
void filter(std::vector<T>& vec, UnaryPredicate pred) {
    vec.erase(std::remove_if(vec.begin(), vec.end(), pred), vec.end());
}