# 面向对象的层
import numpy as np

# 权重矩阵生成函数
def creat_wegihts(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)
# 偏置值生成函数
def creat_biases(n_neurons):
    return np.random.randn(n_neurons)
# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0, inputs)

#  定义一个层类
class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.weights = creat_wegihts(n_inputs, n_neurons)
        self.biases = creat_biases(n_neurons)
    def layer_forward(self, inputs):
        self.output = activation_ReLU(np.dot(inputs, self.weights) + self.biases)
        return self.output




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

# 第一层
layer1 = Layer(2,3)
# weights1 = creat_wegihts(2,3)
# biases1 = creat_biases(3)

# 第二层
layer2 = Layer(3,4)
# weights2 = creat_wegihts(3,4)
# biases2 = creat_biases(4)

# 第三层
layer3 = Layer(4,2)
# weights3 = creat_wegihts(4,2)
# biases3 = creat_biases(2)

# 第一层运算
output1 = layer1.layer_forward(inputs)
# sum1 = np.dot(inputs, weights1) + biases1
# output1 = activation_ReLU(sum1)
print(output1)
print('-------------------')

# 第二层运算
output2 = layer2.layer_forward(output1)
# sum2 = np.dot(output1, weights2) + biases2
# output2 = activation_ReLU(sum2)
print(output2)
print('-------------------')
#
# # 第三层运算
output3 = layer3.layer_forward(output2)
# sum3 = np.dot(output2, weights3) + biases3
# output3 = activation_ReLU(sum3)
print(output3)