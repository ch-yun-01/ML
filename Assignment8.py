import csv
import numpy as np
import matplotlib.pyplot as plt

data = []

with open("./iris.csv", "r") as file:
    reader = csv.reader(file)
    headers = next(reader)
    print(list(headers))
    for row in reader:
        temp = [float(item) for item in row[:4]] + [row[4]]
        data.append(temp)

for i in range(len(data)):
    if data[i][4] == 'Setosa':
        data[i][4] = 0
    elif data[i][4] == 'Versicolor':
        data[i][4] = 1
    else:  # Virginica = 2
        data[i][4] = 2

data = np.array(data)
x = data[:, 0:4]
y = data[:, 4]

# one-hot vector로 만들기
one_hot_y = np.zeros((150, 3))
for i in range(150):
    if y[i] == 0:
        one_hot_y[i] = [1, 0, 0]

    elif y[i] == 1:
        one_hot_y[i] = [0, 1, 0]

    else:
        one_hot_y[i] = [0, 0, 1]


def relu(z):
    return np.maximum(0, z)


def derivative_relu(a_matrix):
    return np.where(a_matrix > 0, 1, 0)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


class NeuralNet:
    def __init__(self, config_layer):
        self.cf = config_layer
        self.w1 = np.random.normal(0, 0.05, size=(self.cf[0], self.cf[1]))
        self.w2 = np.random.normal(0, 0.05, size=(self.cf[1], self.cf[2]))
        self.w3 = np.random.normal(0, 0.05, size=(self.cf[2], self.cf[3]))
        self.w4 = np.random.normal(0, 0.05, size=(self.cf[3], self.cf[4]))
        self.w5 = np.random.normal(0, 0.05, size=(self.cf[4], self.cf[5]))

        self.b1 = np.random.normal(0, 0.05, size=(1, self.cf[1]))
        self.b2 = np.random.normal(0, 0.05, size=(1, self.cf[2]))
        self.b3 = np.random.normal(0, 0.05, size=(1, self.cf[3]))
        self.b4 = np.random.normal(0, 0.05, size=(1, self.cf[4]))
        self.b5 = np.random.normal(0, 0.05, size=(1, self.cf[5]))

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = relu(self.z3)
        self.z4 = np.dot(self.a3, self.w4) + self.b4
        self.a4 = relu(self.z4)
        self.z5 = np.dot(self.a4, self.w5) + self.b5
        self.a5 = softmax(self.z5)
        return self.a5

    def update(self, x, y, lr):
        self.delta5 = self.a5 - y
        self.delta4 = np.dot(self.delta5, self.w5.T) * derivative_relu(self.a4)
        self.delta3 = np.dot(self.delta4, self.w4.T) * derivative_relu(self.a3)  
        self.delta2 = np.dot(self.delta3, self.w3.T) * derivative_relu(self.a2)  
        self.delta1 = np.dot(self.delta2, self.w2.T) * derivative_relu(self.a1)  

        self.w1 = self.w1 - lr * np.dot(x.T, self.delta1) / x.shape[0]
        self.w2 = self.w2 - lr * np.dot(self.a1.T, self.delta2) / x.shape[0]
        self.w3 = self.w3 - lr * np.dot(self.a2.T, self.delta3) / x.shape[0]
        self.w4 = self.w4 - lr * np.dot(self.a3.T, self.delta4) / x.shape[0]
        self.w5 = self.w5 - lr * np.dot(self.a4.T, self.delta5) / x.shape[0]

        self.b1 = self.b1 - lr * np.sum(self.delta1) / x.shape[0]
        self.b2 = self.b2 - lr * np.sum(self.delta2) / x.shape[0]
        self.b3 = self.b3 - lr * np.sum(self.delta3) / x.shape[0]
        self.b4 = self.b4 - lr * np.sum(self.delta4) / x.shape[0]
        self.b5 = self.b5 - lr * np.sum(self.delta5) / x.shape[0]



hidden = [16, 32, 24, 12]
layer = [4] + list(hidden) + [3]
lr = 0.005

nn = NeuralNet(layer)
history = []
loss = 100
iterations = 0

while loss > 0.04:

    output = nn.forward(x)

    loss = cross_entropy_loss(one_hot_y, output)
    history.append(loss)

    nn.update(x, one_hot_y, lr)

    if iterations % 100 == 0:
        print(f"Iteration {iterations}, Loss: {loss}")
    iterations += 1

plt.figure(figsize= (8,6))
plt.plot(history)
plt.title(f'layer : {layer}, lr : {lr}')
plt.show()