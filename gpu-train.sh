#!/usr/bin/env bash
CMD=$1
srun -p csc485 --gres gpu -c 2 python3.12 train.py $1
