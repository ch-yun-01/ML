import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for index in range(4):
    X = np.random.uniform(low=-10.0, high = 10.0, size =30)
    y = np.random.uniform(low=-10.0, high = 10.0, size =30)
    y1, y2 = np.random.randint(-10,10, size = 2)
    a = (y2-y1)/20
    b = y2 - a*10
    plt.subplot(2,2,index+1)
    plt.plot([-10, 10], [y1, y2])
    for i in range(30):
        if a*X[i]+b > y[i]:
            plt.scatter(X[i], y[i], c='r', marker='*')
        else :
            plt.scatter(X[i], y[i], c= 'b', marker ='s')

