# 面向对象的网络
# 面向对象的层
import numpy as np

NETWORK_SHAPE = [2,30,50,2]
# 权重矩阵生成函数
def creat_wegihts(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)
# 偏置值生成函数
def creat_biases(n_neurons):
    return np.random.randn(n_neurons)
# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0, inputs)

# 定义一个层
class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.weights = creat_wegihts(n_inputs, n_neurons)
        self.biases = creat_biases(n_neurons)
    def layer_forward(self, inputs):
        sum1 = np.dot(inputs, self.weights) + self.biases
        self.output = activation_ReLU(sum1)
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
            layer_output = self.layers[i].layer_forward(outputs[i])
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
    # sum1 = np.dot(inputs, weights1) + biases1
    # output1 = activation_ReLU(sum1)
    # print(output1)
    # print('-------------------')
    #
    # # 第二层运算
    # output2 = network.layer[1].layer_forward(output1)
    # # sum2 = np.dot(output1, weights2) + biases2
    # # output2 = activation_ReLU(sum2)
    # print(output2)
    # print('-------------------')
    # #
    # # # 第三层运算
    # output3 = network.layer[2].layer_forward(output2)
    # # sum3 = np.dot(output2, weights3) + biases3
    # # output3 = activation_ReLU(sum3)
    # print(output3)

#-------------------------test-------------------------
def test():
    pass
    # network = Network(NETWORK_SHAPE)
    # print(network.shape)
    # print(network.layers)
    # print(network.layers[0].weights)
    # print(network.layers[0].biases)
    # print(network.layers[1].weights)

#--------------------运行---------------------------
main()