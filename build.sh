#!/bin/bash

my_python="/home/gengjie/anaconda3/bin/python"

compile() {
    local lib_name=$1
    local src_path=$2
    local src_files=$(find ${src_path} -name "*.cpp" ! -name "dllmain.cpp")
    local dst_file="$(pwd)/x64/Release/${lib_name}.dll"
    echo "Compiling ${lib_name} -> ${dst_file} ..."

    g++ ${src_files} -shared -fPIC -std=c++17 -o ${dst_file} $(optimization_flags)
}

optimize() {
    local lib_name=$1
    local src_path=$2
    local src_files=$(find ${src_path} -name "*.cpp" ! -name "dllmain.cpp")
    local dst_file="$(pwd)/x64/Release/${lib_name}.dll"
    local profile_dir="profile_data"

    echo "Compiling ${lib_name} -> ${dst_file} ..."
    g++ -fprofile-generate=${profile_dir} -shared -fPIC -std=c++17 -o ${dst_file} ${src_files} $(optimization_flags)

    # Step 1: Generate performance data
    echo "Generating performance data..."
    timeout 120s ${my_python} single.py

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

if [ ! -d "x64/Release" ]; then mkdir -p x64/Release; fi
rm -f x64/Release

optimize packing5Cpp simulation/packing5Cpp
compile analysisCpp analysis/analysisCpp
