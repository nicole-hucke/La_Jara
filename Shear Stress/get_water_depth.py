#### Average water depth

# Checking working directory
import os
os.getcwd()
working_directory = "C:/Users/bern9483/Documents/Nicole/tracers_data/"
os.chdir(working_directory)

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Creating coordinates
x_coordinate = np.linspace(0, 10, 20)
y_coordinate = 1.5*(x_coordinate - 5)**2 - 25
wse = 5

# Checking cross-section
plt.plot(x_coordinate, y_coordinate, color="k")
plt.axhline(y=5, color="lightblue")
plt.show()

# Defining a function to calculate the average depth
def get_water_depth(x_coordinate, y_coordinate, wse):
    tempCalc1 = wse - y_coordinate
    water_depth = tempCalc1[tempCalc1 > 0]
    water_depth_avg = np.mean(water_depth)
    
    return water_depth_avg, tempCalc1

# Calculating average water depth in my cross-section
aux1, aux2 = get_water_depth(x_coordinate, y_coordinate, wse)