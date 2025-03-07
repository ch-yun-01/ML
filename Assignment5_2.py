import csv
import numpy as np
import matplotlib.pyplot as plt

f = open("advertising.csv", "r")
reader = csv.reader(f)
data = list(reader)

header = data[0]
data = data[1:]
data = np.array(data).astype(float)

def MinMaxScaler(data, column):
      Min = min(data[:,column])
      Max = max(data[:,column])
      for i in range(len(data[:,column])):
            data[i, column] = (data[i, column]-Min)/(Max-Min)

for i in range(4):
    MinMaxScaler(data, i)


X = data[:,0:3]
y= data[:,3]

w = np.zeros(4)
lr = 10**(-7)
epochs=3000000

error_history = []

for epoch in range(epochs):
    y_hat = np.dot(X, w[1:]) + w[0]  
    error = np.mean((y_hat - y) ** 2) / 2
    error_history.append(error)
    gradient = np.zeros_like(w)
    gradient[0] = np.sum(y_hat - y) 
    
    for i in range(1, len(w)):
        gradient[i] = np.dot(X[:, i-1], y_hat - y) 
    
    w -= lr * gradient 
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Error = {error}")

print('w : ', w)

error_history = np.array(error_history)

plt.plot(error_history)
plt.title('MinMaxScaler_all')
plt.show()

plt.plot(np.linspace(10, epochs, epochs-10), error_history[10:epochs])
plt.title('MinMaxScaler_10epochs~')
plt.show()

plt.plot(np.linspace(50000, epochs, epochs-50000), error_history[50000:epochs])
plt.title('MinMaxScaler_50000epochs~')
plt.show()