#pragma once

#include"defs.h"
#include<vector>
#include<functional>
using namespace std;


template<typename a, HashFunc hasher>
int anyHasher(const a& x);

/*
    LookupFunc :: 'a[hashable] -> 'b[any]
    Read the result from a database
*/
template<typename a, typename b, int capacity, HashFunc hasher>
struct LookupFunc
{
    b* data;

    LookupFunc() { ; }
    ~LookupFunc() {
        // never deconstruct
    }
    LookupFunc(b func(const a&), vector<a>& inputs) {
        data = (b*)malloc(capacity * sizeof(b));
        int n = inputs.size();
        for (int i = 0; i < n; i++)
        {
            data[i] = func(inputs[i]);
        }
    }
    LookupFunc(std::function<b(const a&)> func, vector<a>& inputs) {
        data = (b*)malloc(capacity * sizeof(b));
        int n = inputs.size();
        for (int i = 0; i < n; i++)
        {
            data[i] = func(inputs[i]);
        }
    }
    b operator()(const a& x) {
        return data[anyHasher<a, hasher>(x)];
    }
};
