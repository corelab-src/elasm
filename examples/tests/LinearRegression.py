
import hecate as hc
from random import *
import numpy as np
import sys
from pathlib import Path
import time

a_compile_type = sys.argv[1]
a_compile_opt = int(sys.argv[2])


hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

x = [ uniform (-1, 1) for a in range(4096)]
a = 2.0
b = 1.0
y = [ a*point +b + uniform (-0.01, 0.01) for point in x]


# print(res)

W = 1.0
c = 0.0

epochs = 2
learning_rate = -0.01

for i in range(epochs):
    
    error = [ W*x[i]+c-y[i] for i  in range(4096)] 
    errX = [ error[i]* x[i] for i in range(4096)] 
    gradW = sum(errX)/ 2048
    gradb = sum(error)/2048
    Wup = learning_rate * gradW
    bup = learning_rate * gradb
    W = W + Wup
    c = c + bup
        
# print (W, c)

hevm.setInput(0, x)
hevm.setInput(1, y)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() -timer
res = hevm.getOutput()
rms = np.sqrt(np.mean(np.power(res[0] - W, 2) + np.power(res[1] - c, 2)))
print (timer / pow(10,9))
print(rms)

