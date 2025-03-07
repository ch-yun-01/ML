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
x = data[:, 1:3]
y = data[:, -1]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(3)
lr = 10**(-1)
epochs = 100000

X = np.column_stack((np.ones(len(y)), x))

for epoch in range(epochs):
    z = np.dot(X, w)
    y_hat = sigmoid(z)
    error = -np.mean(y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10))
    
    gradient = np.dot(X.T, y_hat-y)
    
    w -= lr * gradient/len(y)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Error = {error}")


print('w : ', w)

plt.figure(figsize=(10, 8))

versicolor_scatter = data[y==0]
virginica_scatter = data[y==1]

plt.scatter(versicolor_scatter[:,1], versicolor_scatter[:, 2], color ='r', marker='*', label = 'Versicolor')
plt.scatter(virginica_scatter[:, 1], virginica_scatter[:, 2], color = 'g', marker='*', label = 'Virginica')

x1_values = np.linspace(min(data[:,1]), max(data[:,1]), 500)
x2_values = (-w[0] - w[1] * x1_values) / w[2]

plt.plot(x1_values, x2_values, color='b')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')
plt.title('Decision Boundary for Versicolor vs. Virginica')
plt.legend(loc='upper left')
plt.show()
