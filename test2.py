# 权重矩阵
# 两个神经元

import numpy as np
a11 = -0.9
a21 = -0.5
a31 = -0.7

a12 = -0.8
a22 = -0.5
a32 = -0.6

a13 = -0.5
a23 = -0.8
a33 = -0.2

inputs = np.array([[a11, a21, a31],
                  [a12, a22, a32],
                  [a13, a23, a33]])

w11 = 0.8
w21 = -0.4
w31 = 0

w12 = 0.7
w22 = -0.6
w32 = 0.2

b1 = np.array([-100,100])
#wegihts = np.array([[w11,w12],
#                    [w21,w22],
#                    [w31,w32]])




# sum1=a1*w1 + a2*w2 + a3*w3 + b1

# 权重矩阵生成函数
def creat_wegihts(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)

# 激活函数
# 偏置值生成函数
def creat_biases(n_neurons):
    return np.random.randn(n_neurons)
def activation_ReLU(inputs):
    return np.maximum(0, inputs)
biases = creat_biases(2)
weights = creat_wegihts(3,2)

sum1 = np.dot(inputs, weights) + biases
# --------------------------
print(weights)
print(sum1)
print('-------------------')
print(activation_ReLU(sum1))
