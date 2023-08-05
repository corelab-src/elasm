import hecate as hc 
import numpy as np
import sys
import torch


def get_flat_weight(file_name):
	x = []
	f = open(file_name,'r')
	for y in f.read().split('\n'):
		if not y == "":
			x.append(float(y))
	return x

#input 784, output 100
def input_to_layer_MNIST(image, W):
    res = [0.00000 for i in range(100)]
    new_W = [[0.00000 for j in range(800)] for i in range (100)]
    for n in range(100) :
        for c in range(8) :
            for k in range(100) :
                index = c*100 + k
                if(index < 784) : 
                    if (index+n >= 784) :
                        new_W[n][800-n+(index+n)%784] = W[100-n+(index+n)%784][(index+n)%784]
                    else :
                        new_W[n][index] = W[k][(index + n) % 784 ]
    new_W = [hc.Plain(Win) for Win in new_W]
    for n in range(100) :
        rot = image.rotate(n)
        mul = rot * new_W[n]
        result = mul if n == 0 else result + mul
     
    m = 800
    res = result
    for i in range(3):
        m = m >> 1
        temp = res.rotate(m)
        res = res + temp
    
    return res

#input 100, output 10
def layer_to_output_MNIST(image, W):
    res = [0.0 for i in range(10)]
    new_W = [[0.0 for j in range(100)] for i in range (10)]
    for n in range(10) :
        for c in range(10) :
            for k in range(10) :
                index = c *10 + k
                if(c * 10 + k < 100) :
                    new_W[n][index] = W[k][(index + n) % 100 ]
    new_W = [hc.Plain(Win) for Win in new_W]
    for n in range(10):
        rot = image.rotate(n)
        mul = rot * new_W[n]
        result = mul if n == 0 else result + mul
     
    temp = result.rotate(50)
    res = result + temp
    for i in range(5):
        temp = res.rotate(i*10)
        res1 = temp if i == 0 else res1 + temp
    return res1



@hc.func("c")
def MLP(image) :
    from pathlib import Path

    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    model = torch.load(str(source_dir)+"/../data/mlp.model", map_location=torch.device('cpu'))
    W1 = model["linear1.weight"].cpu().detach().numpy()
    b1 = model["linear1.bias"].cpu().detach().numpy()
    W2 = model["linear2.weight"].cpu().detach().numpy()
    b2 = model["linear2.bias"].cpu().detach().numpy()
    h1 = input_to_layer_MNIST(image, W1) + hc.Plain(list(b1))
    h = h1 * h1
    res = layer_to_output_MNIST(h, W2) + hc.Plain(list(b2))
    return res



modName = hc.save("traced", "traced")
print (modName)


