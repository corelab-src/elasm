import hecate as hc 
import sys
import numpy as np


def roll(a, i) : 
    return a.rotate(i)

@hc.func("c")
def SobelFilter(lena_array) :

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
    d = c*c*c *0.173 - c * c *1.098 +  c * 2.214 
    return d


modName = hc.save("traced", "traced")
print (modName)
