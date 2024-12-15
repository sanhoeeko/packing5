#!/bin/bash

compile() {
    local lib_name=$1
    local src_path=$2
    local src_files="${src_path}/*.cpp"
    local dst_file="x64/${lib_name}.so"

    g++ ${src_files} -shared -fPIC -std=c++17 -o ${dst_file} $(optimization_flags)
}

optimize() {
    local lib_name=$1
    local src_path=$2
    local src_files="${src_path}/*.cpp"
    local dst_file="x64/${lib_name}.so"
    local profile_dir="profile_data"

    # Step 1: Generate performance data
    echo "Generating performance data..."
    g++ -fprofile-generate=${profile_dir} -shared -fPIC -std=c++17 -o ${dst_file} ${src_files} $(optimization_flags)
    timeout 120s python3 single.py

    # Step 2: Use performance data for optimization
    echo "Using performance data for optimization..."
    g++ -fprofile-use=${profile_dir} -shared -fPIC -std=c++17 -o ${dst_file} ${src_files} $(optimization_flags)

    echo "Optimization complete. Shared library ${lib_name} has been optimized."
}

optimization_flags() { 
    echo "-fopenmp -march=native -Wall -Ofast -funroll-loops -flto" 
}

# build DLL
cd program
rm -f x64
optimize packing5Cpp simulation/packing5Cpp
compile analysisCpp analysis/analysisCpp
