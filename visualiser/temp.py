import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# create 2 kernels
m1 = (50,50)
s1 = np.eye(2)*400
k1 = multivariate_normal(mean=m1, cov=s1)

m2 = (150,150)
s2 = np.eye(2)*400
k2 = multivariate_normal(mean=m2, cov=s2)

# create a grid of (x,y) coordinates at which to evaluate the kernels


x = np.arange(0,200)
y = np.arange(0,200)
xx, yy = np.meshgrid(x,y)

# evaluate kernels at grid points
xxyy = np.c_[xx.ravel(), yy.ravel()]

zz = np.zeros((xxyy.shape[0])).reshape(200,200)

x1lim = (0,100)
y1lim = (0,100)
x1res = x1lim[1] - x1lim[0]
y1res = x1res
x1 = np.linspace(x1lim[0], x1lim[1])
y1 = np.linspace(y1lim[0], y1lim[1])
xx1, yy1 = np.meshgrid(x1,y1)

# evaluate kernels at grid points
xxyy1 = np.c_[xx1.ravel(), yy1.ravel()]

zz1 = k1.pdf(xxyy1) + k2.pdf(xxyy1)
zz1 = zz1.reshape(50,50)
zz[100:150, 50:100] += zz1
plt.imshow(zz1); plt.show()