import numpy as np

a11 = 0
a21 = 1

a12 = 0.5
a22 = 0.5

a13 = 0.2
a23 = 0.8

a14 = 0.7
a24 = 0.3

a15 = 0.9
a25 = 0.1
# batch
predicted = np.array([[a11, a21],
                      [a12, a22],
                      [a13, a23],
                      [a14, a24],
                      [a15, a25]])
real = np.array([1,0,1,0,1])
print(predicted)

# 需求函数
def get_final_layer_preAct_damand(predicted_values, target_vector):
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector

    for i in range(len(target_vector)):
        if np.dot(target[i], predicted_values[i]) > 0.5:
            target[i] = np.array([0,0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target
print(get_final_layer_preAct_damand(predicted, real))