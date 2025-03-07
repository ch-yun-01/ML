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
    if data[i][4] =='Versicolor':
        data[i][4] = 0
    elif data[i][4] == 'Virginica':
        data[i][4] = 1
    else :
        data[i][4] = None

filtered_data = [row for row in data if None not in row]
data = np.array(filtered_data)
x = data[:, 0:4]
y = data[:, 4]
x = x.T
y = y.T

#시그모이드 함수
def sigmoid(z):
        return 1 / (1 + np.exp(-z))

#시그모이드 도함수
def sigmoid_derivative(a):
        return a * (1 - a)


class NeuralNet: 
    def __init__(self, config_layer):
        self.cf = config_layer
        self.w1 = np.random.rand(self.cf[1], self.cf[0])
        self.w2 = np.random.rand(self.cf[2], self.cf[1])
        self.w3 = np.random.rand(self.cf[3], self.cf[2])
        self.b1 = np.random.rand(self.cf[1], 1)
        self.b2 = np.random.rand(self.cf[2], 1)
        self.b3 = np.random.rand(self.cf[3], 1)

    def forward(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.w3, self.a2) + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def update(self, x, lr, y):
        self.delta3 = -(y - self.a3)
        self.delta2 = np.dot(self.w3.T, self.delta3) * sigmoid_derivative(self.a2)
        self.delta1 = np.dot(self.w2.T, self.delta2) * sigmoid_derivative(self.a1)
        self.w1 = self.w1-lr*np.dot(self.delta1, x.T)/x.shape[1]
        self.w2 = self.w2-lr*np.dot(self.delta2, self.a1.T)/x.shape[1]
        self.w3 = self.w3-lr*np.dot(self.delta3, self.a2.T)/x.shape[1]
        self.b1 = self.b1-lr*np.sum(self.delta1)/x.shape[1]
        self.b2 = self.b2-lr*np.sum(self.delta2)/x.shape[1]
        self.b3 = self.b3-lr*np.sum(self.delta3)/x.shape[1]

np.where
#그래프 그리기
def draw_plot(w, b, a2, output, iteration, loss):
    plt.figure(figsize=(5,4))
    a2_1 = a2[0]
    a2_2 = a2[1]
    
    for i in range(len(output[0])):
        if output[0][i] >=0.5:
            plt.scatter(a2_1[i], a2_2[i], color = 'r', marker='*')

        else : 
            plt.scatter(a2_1[i], a2_2[i], color = 'g', marker='*')
    
    a2_1_values = np.linspace(min(a2_1), max(a2_1), 100)
    a2_2_values = (-b[0] - w[0][0] * a2_1_values) / w[0][1]

    
    plt.plot(a2_1_values, a2_2_values, color='b')
    plt.title(f"Iteration {iteration}, Loss: {loss}")
    plt.xlabel('a2_1')
    plt.ylabel('a2_2')
    plt.legend()
    plt.show()


layer_config = [4, 3, 2, 1]
model = NeuralNet(layer_config)

iterations = 5000000
learning_rate = 0.04

for i in range(iterations):
    
    output = model.forward(x)
    model.update(x, learning_rate, y)
    loss = np.mean(-y * np.log(output) - (1 - y) * np.log(1 - output))

    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss}")
        if i % 50000 == 0:
            draw_plot(model.w3, model.b3, model.a2, model.a3, i, loss)
            
    if loss < 10**(-3):
        draw_plot(model.w3, model.b3, model.a2, model.a3, i, loss)
        break

