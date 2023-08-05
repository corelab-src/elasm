
import hecate as hc
import numpy as np


def roll (a, i) :
    return np.roll(a, -i)

def preprocess():

    lena = Image.open(f'{hc.hecate_dir}/examples//data/cornertest.jpg').convert('L')
    lena = lena.resize((64,64))
    lena_array = np.asarray(lena.getdata(), dtype=np.float64) / 256
    lena_array = lena_array.reshape([64*64])

    return lena_array.reshape([1, 4096]);

def process(lena_array) :

    lena_array = lena_array[0]
    F = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ix = 0
    Iy = 0
    for i in range(3) : 
        for j in range(3) : 
            rot = roll (lena_array, i*64 +j)
            h = rot * F[i][j]
            v = rot * F[j][i]
            Ix = Ix + h
            Iy = Iy + v
    Ix2 = Ix * Ix 
    Iy2 = Iy * Iy 
    c = Ix2 + Iy2
    d = 0.173*c*c*c - 1.098 * c * c + 2.214*c 

    return d.reshape([1, 4096])

def postprocess (result) : 
    return (result *256) [:, :4096]

## EVAL

if __name__ == "__main__" :

    from random import *
    import sys
    from pathlib import Path
    import time 
    from PIL import Image

    a_compile_type = sys.argv[1]
    a_compile_opt = int(sys.argv[2])
    hevm = hc.HEVM()
    stem = Path(__file__).stem
    hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

    input_dat = preprocess()
    reference = postprocess(process(input_dat))
    [hevm.setInput(i, dat) for i, dat in enumerate(input_dat)]
    timer = time.perf_counter_ns()
    hevm.run()
    timer = time.perf_counter_ns() -timer
    res = hevm.getOutput()
    res = postprocess(res)
    err = res - reference 
    rms = np.sqrt( np.sum(err*err) / res.shape[-1])
    print (timer/ (pow(10, 9)))
    print (rms)



