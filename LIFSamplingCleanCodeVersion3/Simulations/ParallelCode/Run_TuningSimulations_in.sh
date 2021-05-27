#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:16:42 2020

@author: achattoraj
"""

#!/bin/bash



#SBATCH -o Run_TuningSimulations_in_log.txt -t 20:00:00
#SBATCH --mem-per-cpu=5gb
#SBATCH -n 25
#SBATCH -J Ankani

module load python/2.7.12 brian/2.0

python Run_TuningSimulations_in.py $SLURM_ARRAY_TASK_ID