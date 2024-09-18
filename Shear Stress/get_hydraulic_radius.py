#### Average water depth

# Checking working directory
import os
os.getcwd()
working_directory = "C:/Users/bern9483/Documents/Nicole/tracers_data/"
os.chdir(working_directory)

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

#### Defining some functions
def get_hydraulic_radius(X, Y, WSE):
    wet_areas = []
    wet_perimeters = []
    poly_x = []
    poly_y = []

    # Loop through the points of the cross-section to find wet polygon
    for i in range(len(X) - 1):
        y1, y2 = Y[i], Y[i+1]
        x1, x2 = X[i], X[i+1]

        # First, checking if we enter the water
        if (y1 >= WSE and y2 < WSE):
            # Linear interpolation to find the intersection point
            x_int1 = x1 + (WSE - y1) * (x2 - x1) / (y2 - y1)
            poly_x.append(x_int1)
            poly_y.append(WSE)
        
        # If not, we check if the first point is below the WSE
        if (y1 < WSE):
            poly_x.append(x1)
            poly_y.append(y1)
        
        # Then, we check if we go out of the water. If we did, we calculate the area and store it
        if (y1 < WSE and y2 >= WSE):
            # Linear interpolation to find the intersection point
            x_int2 = x1 + (WSE - y1) * (x2 - x1) / (y2 - y1)
            poly_x.append(x_int2)
            poly_y.append(WSE)

            # Closing the polygon
            if len(poly_x) > 0:
                poly_x.append(poly_x[0])
                poly_y.append(poly_y[0])
            
            # Calculating the area
            tempCalc1 = 0.5 * np.abs(np.dot(poly_x, np.roll(poly_y, 1)) - np.dot(poly_y, np.roll(poly_x, 1)))
            wet_areas.append(tempCalc1)

            # Calculating the wet perimeter
            distances = np.sqrt(np.diff(poly_x)**2 + np.diff(poly_y)**2)
            wet_perimeters.append(np.sum(distances[:-1]))

            # Resetting the polygon
            poly_x = []
            poly_y = []
    
    # return hydraulic radius, wet area, wet perimeter
    tempCalc1 = np.sum(wet_areas)
    tempCalc2 = np.sum(wet_perimeters)
    hydraulic_radius = tempCalc1/tempCalc2
    
    return hydraulic_radius

# EXAMPLE 1
# Creating coordinates
X = np.linspace(0, 17, 2000)
Y = 2*np.sin(X)
WSE = 0

# Closing the cross-section to avoid overbank flow
X = np.insert(X, 0, X[0] - 0.5)
Y = np.insert(Y, 0, WSE + 1)

X = np.append(X, X[-1] + 0.5)
Y = np.append(Y, WSE + 1)

# Checking cross-section
plt.plot(X, Y, color="k")
plt.axhline(y=WSE, color="lightblue")
plt.show()

tempCalc1 = get_hydraulic_radius(X, Y, WSE)
tempCalc1
