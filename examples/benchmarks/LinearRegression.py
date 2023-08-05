import hecate as hc
import sys


def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1<<(11-i))
        data = data +rot

    return data

@hc.func("c,c")
def LinearRegression(x_data, y_data) :
    W = hc.Plain([1.0])
    b = hc.Plain([0.0])
    
    epochs = 2
    learning_rate = hc.Plain([-0.01])

    for i in range(epochs):
        xW = x_data*W
        xWb = xW + b

        error = xWb - y_data
        
        errX = error * x_data
        meanErrX = errX * hc.Plain([1/2048])
        gradW = sum_elements(meanErrX)
        meanErr = error * hc.Plain([1/2048])
        gradb = sum_elements(meanErr)
        Wup = learning_rate * gradW
        bup = learning_rate * gradb
        W = W + Wup
        b = b + bup

    return W, b



modName = hc.save("traced", "traced")
print (modName)


