import csv
import numpy as np
import matplotlib.pyplot as plt

f = open("advertising.csv", "r")
reader = csv.reader(f)
data = list(reader)

header = data[0]
tv_index = header.index('TV')
sales_index = header.index('Sales')

X = [float(row[tv_index]) for row in data[1:]]
y = [float(row[sales_index]) for row in data[1:]]

X = np.array(X)
y = np.array(y)

w1 = 0.0
w0 = 0.0
lr = 10**(-6)
iteration = 10900000

for iter in range(iteration):
    index = np.random.randint(len(X))
    X_temp = X[index]
    y_temp = y[index]

    y_hat = w0 + w1 * X_temp
    w0 -= lr * (y_hat - y_temp)
    w1 -= lr * X_temp * (y_hat - y_temp)

    error = ((y - (w0 + w1 * X)) ** 2).mean()

    if iter % 10 == 0:
        print(f"iteration {iter}: Error = {error}")


y_pred = w1*X+w0
plt.scatter(X,y)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('Assignment4_1')
plt.plot(X,y_pred,'r')
plt.show()
