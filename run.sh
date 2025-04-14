#!/bin/bash

PYTHON="/home/gengjie/anaconda3/bin/python"

rm nohup.out

# Simulation and pack
nohup ${PYTHON} program/main.py &
wait

nohup ${PYTHON} program/shell.py auto_pack &
wait

today=$(date +%Y%m%d)
mv data.h5 "data-${today}.h5"

# Analysis
nohup ${PYTHON} program/shell.py analysis "data-${today}.h5" &
wait
