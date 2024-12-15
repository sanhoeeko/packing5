#!/bin/bash

my_python="/home/gengjie/anaconda3/bin/python"

rm nohup.out
nohup ${my_python} program/main.py &
wait
