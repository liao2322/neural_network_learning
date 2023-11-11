# softmax激活函数
import numpy as np
def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True) #保持之前的形状不变
    slided_inputs = inputs - max_values
    print(slided_inputs)
    exp_values = np.exp(slided_inputs)
    print(exp_values)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    print(norm_base)
    norm_values = exp_values / norm_base
    print(norm_values)
    return norm_values
a11 = -0.9
a21 = -0.5
a31 = -0.7

a12 = -0.8
a22 = -0.5
a32 = -0.6

a13 = -0.5
a23 = -0.8
a33 = -0.2

a14 = 0.5
a24 = 0.8

a15 = 0.5
a25 = 0.8

# batch
inputs = np.array([[a11,a21],
                   [a12,a22],
                   [a13,a23],
                   [a14,a24],
                   [a15,a25]])
print(inputs)
print(activation_softmax(inputs))