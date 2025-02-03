# test nics

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# creating variables
x0 = np.arange(0,10)
y0 = 10 + np.sin(x0)

# just checking
plt.plot(x0, y0)
plt.show()

# defining function
def trapz2(x0, y0):
    sub_area = []
    for i in range(len(x0) - 1):
        xpair = x0[i:i+2]
        ypair = y0[i:i+2]
        tempCalc1 = np.trapz(xpair, ypair)
        tempCalc1 = np.abs(tempCalc1)
        sub_area = np.append(sub_area, tempCalc1)

    return np.sum(sub_area)

area = trapz2(x0, y0)
