# 不同形状数组相加

import numpy as np
def creat_biases(n_neurons):
    return np.random.randn(n_neurons)
a1 = np.array([[1,2]
              ,[3,4]])
a2 = np.array([[-1],[1]])
a3 = np.array([-1,1])

biases = creat_biases(2)
print(a1 + a2)
print(a1 + a3)
print(biases)