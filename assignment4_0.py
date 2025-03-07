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
lr = 10**(-6.5)
epochs = 1150000
for epoch in range(epochs):
    y_hat = X*w1 + w0
    error = ((y - (w0 + w1 * X)) ** 2).mean()

    w1 = w1 - lr * np.sum((y_hat - y) * X)
    w0 = w0 - lr * np.sum(y_hat - y)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Error = {error}")

y_pred = w1*X+w0
plt.scatter(X,y)
plt.plot(X,y_pred,'r')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('Assignment4_0')
plt.show()

