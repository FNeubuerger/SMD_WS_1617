# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)

xx, yy = np.meshgrid(x, y)

# Get the 2d Gauß values
XX = xx.flatten()
YY = yy.flatten()

# Scipy needs the x, y values as: [[x1, y1], [x2, y2], ..., [xN, yN]]
vals = list(zip(XX, YY))
gaus_pdf = scs.multivariate_normal.pdf(
    vals, mean=[0, 0], cov=[[1, 0.5], [0.5, 2]])

# Now reshape for pcolormesh
zz = gaus_pdf.reshape(xx.shape)

# And plot it
plt.pcolormesh(xx, yy, zz, cmap="pink")
plt.savefig("2dgaus.png", dpi=150)
