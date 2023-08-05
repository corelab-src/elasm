import hecate as hc
import sys
# import pandas as pd
import torch
from torchvision import datasets, transforms


from PIL import Image
import numpy as np
from random import *
import pprint


from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent

def preprocess():
    x = [ uniform (0.0, 1.0) for a in range(784)]
    b = [ 0 for a in range(16)]
    x = x+b
    return np.array(x)

def process(x) : 
    model = torch.load(str(source_dir)+"/../data/mlp.model", map_location=torch.device('cpu'))
    W1 = model["linear1.weight"].cpu().detach().numpy()
    b1 = model["linear1.bias"].cpu().detach().numpy()
    W2 = model["linear2.weight"].cpu().detach().numpy()
    b2 = model["linear2.bias"].cpu().detach().numpy()

    inter = [0.0 for i in range (100)]
    res = [0.0 for i in range (10)]

    for i in range(100) : 
        for j in range(784) :
            inter[i] += x[j] * W1[i][j]
        inter[i] += b1[i]
    inter = [i*i for i in inter]
    for i in range(10) : 
        for j in range(100) :
            res[i] += inter[j] * W2[i][j]
        res[i] += b2[i]
    
    return np.array([res])

def postprocess(res) : 
    return res[0,:10]



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
    [hevm.setInput(i, dat) for i, dat in enumerate([input_dat])]
    timer = time.perf_counter_ns()
    hevm.run()
    timer = time.perf_counter_ns() -timer
    res = hevm.getOutput()
    res = postprocess(res)
    err = res - reference 
    rms = np.sqrt( np.sum(err*err) / res.shape[-1])
    print (timer/ (pow(10, 9)))
    print (rms)
