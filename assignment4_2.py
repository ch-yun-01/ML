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

w0_values = np.linspace(-10, 10, 200)
w1_values = np.linspace(-10, 10, 200)
w0_grid, w1_grid = np.meshgrid(w0_values, w1_values)
loss_values = np.zeros_like(w0_grid)

for i in range(len(w0_values)):
    for j in range(len(w1_values)):
        y_pred = w0_grid[i, j] + X*w1_grid[i,j]
        loss_values[i, j] = np.sum((y_pred-y)**2)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w0_grid, w1_grid, loss_values, cmap='plasma')
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface')
ax.view_init(10,25)

plt.show()

w0_values = np.linspace(-100, 100, 200)
w1_values = np.linspace(-1, 1, 200)
w0_grid, w1_grid = np.meshgrid(w0_values, w1_values)
loss_values = np.zeros_like(w0_grid)

for i in range(len(w0_values)):
    for j in range(len(w1_values)):
        y_pred = w0_grid[i, j] + X*w1_grid[i,j]
        loss_values[i, j] = np.sum((y_pred-y)**2)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w0_grid, w1_grid, loss_values, cmap='plasma')
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface')
ax.view_init(10,5)

plt.show()

