# 面向对象的网络
# 面向对象的层
import numpy as np

NETWORK_SHAPE = [2,3,5,2]
# 权重矩阵生成函数
def creat_wegihts(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)
# 偏置值生成函数
def creat_biases(n_neurons):
    return np.random.randn(n_neurons)
# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0, inputs)
# softmax激活函数
def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True) #保持之前的形状不变
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    norm_values = exp_values / norm_base
    return norm_values

# 定义一个层
class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.weights = creat_wegihts(n_inputs, n_neurons)
        self.biases = creat_biases(n_neurons)
    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
 #       self.output = activation_ReLU(sum1)
        return self.output
# 定义一个网络类
class Network():
    def __init__(self, NETWORK_SHAPE):
        self.shape = NETWORK_SHAPE
        self.layers = []
        for i in range(len(NETWORK_SHAPE)-1):
            layer = Layer(NETWORK_SHAPE[i], NETWORK_SHAPE[i+1])
            self.layers.append(layer)
            # 前馈运算函数
    def network_forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers)-1:
                layer_output = activation_ReLU(layer_sum)
            else:
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
        print(outputs)
        return outputs



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


def main():
    network = Network(NETWORK_SHAPE)
    network.network_forward(inputs)

#-------------------------test-------------------------
def test():
    pass

#--------------------运行---------------------------
main()