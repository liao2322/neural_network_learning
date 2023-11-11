# test
# 一个神经元
import numpy as np
a1 = -0.9
a2 = -0.5
a3 = -0.7

inputs = np.array([a1, a2, a3],)

w1 = 0.8
w2 = -0.4
w3 = 0
b1 = 0
wegihts = np.array([[w1],
                    [w2],
                    [w3]])

sum1 = np.dot(inputs, wegihts) + b1

# sum1=a1*w1 + a2*w2 + a3*w3 + b1

# 激活函数

def activation_ReLU(inputs):
    return np.maximum(0, inputs)

print(sum1)
print(activation_ReLU(sum1))
