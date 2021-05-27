import os
import sys

#f="mpirun -np 4 python run-em.py params-20x20-dog/H_test_million/bsc_test_h128_iter_"+sys.argv[1]+".py"
f="python run-em.py params-20x20-dog/H_test/bsc_test"+sys.argv[1]+".py"
#f="python try.py"
os.system(f)