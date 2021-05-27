#!/bin/bash


## #SBATCH -o h128_log.%a.txt -t 10:01:00
#SBATCH -o h64_log.%a.txt -t 10:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1 
#SBATCH -J Ankani
#SBATCH -a 1,2

module load python/2.7.12 openmpi/2.1.1/b1

mpirun -n 8 python h64_init_W.py $SLURM_ARRAY_TASK_ID