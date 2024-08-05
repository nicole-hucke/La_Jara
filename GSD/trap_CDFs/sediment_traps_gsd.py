###############################################################
# Importing libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Defining some functions

# Removing outlier GSDs
def remove_outlier_gsd(matrix):
    # Calculating representative sizes
    matrix_d16 = np.percentile(matrix, 16, axis=0)
    matrix_d50 = np.percentile(matrix, 50, axis=0)
    matrix_d84 = np.percentile(matrix, 84, axis=0)
    # Calculating Quartile 1
    tempCalc1 = np.percentile(matrix_d16, 25)
    tempCalc2 = np.percentile(matrix_d50, 25)
    tempCalc3 = np.percentile(matrix_d84, 25)
    matrix_q1 = np.array((tempCalc1, tempCalc2, tempCalc3)) # d16, d50, d84
    # Calculating Quartile 3
    tempCalc1 = np.percentile(matrix_d16, 75)
    tempCalc2 = np.percentile(matrix_d50, 75)
    tempCalc3 = np.percentile(matrix_d84, 75)
    matrix_q3 = np.array((tempCalc1, tempCalc2, tempCalc3)) # d16, d50, d84
    # Calculating IQR and lower/upper bounds
    spring_wc_IQR = matrix_q3 - matrix_q1
    lower_bound = matrix_q1 - 1.5*spring_wc_IQR
    upper_bound = matrix_q3 + 1.5*spring_wc_IQR
    # Identifying GSDs with representative sizes outside the range
    tempBool1 = np.where(matrix_d16 < lower_bound[0], True, False)
    tempBool2 = np.where(matrix_d50 < lower_bound[1], True, False)
    tempBool3 = np.where(matrix_d84 < lower_bound[2], True, False)
    tempBool1 + tempBool2 + tempBool3
    np.arange(len(tempBool1))[tempBool1 + tempBool2 + tempBool3]
    # Identifying GSDs with representative sizes outside the range
    tempBool4 = np.where(matrix_d16 > upper_bound[0], True, False)
    tempBool5 = np.where(matrix_d50 > upper_bound[1], True, False)
    tempBool6 = np.where(matrix_d84 > upper_bound[2], True, False)
    tempBool4 + tempBool5 + tempBool6
    np.arange(len(tempBool4))[tempBool4 + tempBool5 + tempBool6]
    # Deleting outlier GSDs
    tempCalc1 = np.arange(len(tempBool1))[tempBool1 + tempBool2 + tempBool3 + tempBool4 + tempBool5 + tempBool6]
    tempCalc2 = np.delete(matrix, tempCalc1, axis=1)

    return tempCalc2

######################################################################################

# Checking environment and working directory
import os
os.getcwd()
working_directory = "C:/Users/bern9483/Documents/Nicole/sediment_traps_gsd/"
os.chdir(working_directory)

# Filenames and directories
results_directory = "./plots/"
filename1 = "./Spring_2023_watercolumn_percentage.csv"
filename2 = "./Spring_2023_basket_percentage.csv"
filename3 = "./Summer_2023_watercolumn_percentage.csv"
filename4 = "./Summer_2023_basket_percentage.csv"

# Reading files
dataframe1 = pd.read_csv(filename1, skiprows=5)
dataframe2 = pd.read_csv(filename2, skiprows=5)
dataframe3 = pd.read_csv(filename3, skiprows=5)
dataframe4 = pd.read_csv(filename4, skiprows=5)

# Getting information
grain_sizes = np.array(dataframe1.iloc[:-1, 0])
proportion1 = np.array(dataframe1.iloc[:-1, 1:])
proportion2 = np.array(dataframe2.iloc[:-1, 1:])
proportion3 = np.array(dataframe3.iloc[:, 1:])
proportion4 = np.array(dataframe4.iloc[:-1, 1:])

# Defining open/closed baskets
spring_st = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
summer_st = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])

# Calculating cumulative sum
spring_wc_gsd = np.cumsum(proportion1, axis=0)
spring_st_gsd = np.cumsum(proportion2, axis=0)
summer_wc_gsd = np.cumsum(proportion3, axis=0)
summer_st_gsd = np.cumsum(proportion4, axis=0)

# Removing outlier GSDs
spring_wc_gsd = remove_outlier_gsd(spring_wc_gsd)
summer_wc_gsd = remove_outlier_gsd(summer_wc_gsd)

# Calculating max, avg and min GSDs
spring_wc_gsd_max = np.max(spring_wc_gsd, axis=1)
spring_wc_gsd_avg = np.mean(spring_wc_gsd, axis=1)
spring_wc_gsd_min = np.min(spring_wc_gsd, axis=1)
summer_wc_gsd_max = np.max(summer_wc_gsd, axis=1)
summer_wc_gsd_avg = np.mean(summer_wc_gsd, axis=1)
summer_wc_gsd_min = np.min(summer_wc_gsd, axis=1)

# Plotting SUMMER
#colormap1 = cm.get_cmap('Oranges', summer_st_gsd.shape[1])
#colors1 = colormap1(np.linspace(0,1, summer_st_gsd.shape[1]))
color_closed ="lightslategrey"
color_open = "darkorange"
basket_closed = 20
basket_open = 21
tempString1 = "Grain size distribution\nSummer: T8-C and T8-D"
outname = "summer-t8c-t8d"
distance_closed = wasserstein_distance(summer_st_gsd[:, basket_closed], summer_wc_gsd_avg)
distance_open = wasserstein_distance(summer_st_gsd[:, basket_open], summer_wc_gsd_avg)
distance_closed = np.round(distance_closed, 4)
distance_open = np.round(distance_open, 4)
rmse_closed = np.mean((summer_st_gsd[:, basket_closed] - summer_wc_gsd_avg)**2)
rmse_open = np.mean((summer_st_gsd[:, basket_open] - summer_wc_gsd_avg)**2)
rmse_closed = np.round(rmse_closed, 5)
rmse_open = np.round(rmse_open, 5)
plt.figure(figsize=(9, 8))
plt.plot(grain_sizes, summer_st_gsd[:, basket_closed], color=color_closed, label="Closed Basket")
plt.plot(grain_sizes, summer_st_gsd[:, basket_open], color=color_open, label="Open Basket")
plt.plot(grain_sizes, summer_wc_gsd_max, color="black", linestyle="--", alpha=0.25)
plt.plot(grain_sizes, summer_wc_gsd_avg, color="black", label="Water Column")
plt.plot(grain_sizes, summer_wc_gsd_min, color="black", linestyle="--", alpha=0.25)
tempString2 = "Wasserstein Distance\nOB: "+str(distance_open) +"\nCB: "+str(distance_closed) +\
                "\nRMSE\nOB: "+str(rmse_open) +"\nCB: "+str(rmse_closed)
plt.text(0.95, 0.05, tempString2, fontsize=10,
         horizontalalignment='right', verticalalignment='bottom',
         transform=plt.gca().transAxes)
plt.xlabel("Grain size, Di ($\mu$m)")
plt.ylabel("% percent finer")
plt.title(tempString1)
plt.grid(True)
plt.legend()
plt.xscale('log')
plt.xlim((0.1, 1000))
plt.savefig(results_directory + outname + ".png", dpi=300)
plt.show()

# Plotting SPRING
#colormap2 = cm.get_cmap('Blues', spring_st_gsd.shape[1])
#colors2 = colormap2(np.linspace(0,1, spring_st_gsd.shape[1]))
color_closed ="lightslategrey"
color_open = "dodgerblue"
basket_closed = 21
basket_open = 22
tempString1 = "Grain size distribution\nSpring: T8-C and T8-D"
outname = "spring-t8c-t8d"
distance_closed = wasserstein_distance(spring_st_gsd[:, basket_closed], spring_wc_gsd_avg)
distance_open = wasserstein_distance(spring_st_gsd[:, basket_open], spring_wc_gsd_avg)
distance_closed = np.round(distance_closed, 4)
distance_open = np.round(distance_open, 4)
rmse_closed = np.mean((spring_st_gsd[:, basket_closed] - spring_wc_gsd_avg)**2)
rmse_open = np.mean((spring_st_gsd[:, basket_open] - spring_wc_gsd_avg)**2)
rmse_closed = np.round(rmse_closed, 5)
rmse_open = np.round(rmse_open, 5)
plt.figure(figsize=(9, 8))
plt.plot(grain_sizes, spring_st_gsd[:, basket_closed], color=color_closed, label="Closed Basket")
plt.plot(grain_sizes, spring_st_gsd[:, basket_open], color=color_open, label="Open Basket")
plt.plot(grain_sizes, spring_wc_gsd_max, color="black", linestyle="--", alpha=0.25)
plt.plot(grain_sizes, spring_wc_gsd_avg, color="black", label="Water Column")
plt.plot(grain_sizes, spring_wc_gsd_min, color="black", linestyle="--", alpha=0.25)
tempString2 = "Wasserstein Distance\nOB: "+str(distance_open) +"\nCB: "+str(distance_closed) +\
                "\nRMSE\nOB: "+str(rmse_open) +"\nCB: "+str(rmse_closed)
plt.text(0.95, 0.05, tempString2, fontsize=10,
         horizontalalignment='right', verticalalignment='bottom',
         transform=plt.gca().transAxes)
plt.xlabel("Grain size, Di ($\mu$m)")
plt.ylabel("% percent finer")
plt.title(tempString1)
plt.grid(True)
plt.legend()
plt.xscale('log')
plt.xlim((0.1, 1000))
plt.savefig(results_directory + outname + ".png", dpi=300)
plt.show()

##################
# Creating colormaps
colormap1_summer = cm.get_cmap('Greys', np.sum(summer_st == 0))
colormap2_summer = cm.get_cmap('Greens', np.sum(summer_st == 1))

colormap1_spring = cm.get_cmap('Greys', np.sum(spring_st == 0))
colormap2_spring = cm.get_cmap('Blues', np.sum(spring_st == 1))

# Picking colors
colors1_summer = colormap1_summer(np.linspace(0, 1, np.sum(summer_st == 0)))
colors2_summer = colormap2_summer(np.linspace(0, 1, np.sum(summer_st == 1)))

colors1_spring = colormap1_spring(np.linspace(0, 1, np.sum(spring_st == 0)))
colors2_spring = colormap2_spring(np.linspace(0, 1, np.sum(spring_st == 1)))

colors_summer = []
colors1_index = 0
colors2_index = 0
for value in summer_st:
    if value == 0:
        #colors_summer.append(colors1_summer[colors1_index])
        colors_summer.append("lightslategrey")
        colors1_index += 1
    else:
        #colors_summer.append(colors2_summer[colors2_index])
        colors_summer.append("darkorange")
        colors2_index += 1

colors_spring = []
colors1_index = 0
colors2_index = 0
for value in spring_st:
    if value == 0:
        #colors_spring.append(colors1_spring[colors1_index])
        colors_spring.append("grey")
        colors1_index += 1
    else:
        #colors_spring.append(colors2_spring[colors2_index])
        colors_spring.append("dodgerblue")
        colors2_index += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
# Plotting SUMMER
#colormap = cm.get_cmap('Oranges', summer_st_gsd.shape[1])
#colors = colormap(np.linspace(0, 1, summer_st_gsd.shape[1]))
ax1.set_prop_cycle(color=colors_summer)
ax1.plot(grain_sizes, summer_st_gsd)
ax1.plot(grain_sizes, summer_wc_gsd_max, color="darkred", linestyle="--", alpha=0.25)
ax1.plot(grain_sizes, summer_wc_gsd_avg, color="darkred", label="Water column")
ax1.plot(grain_sizes, summer_wc_gsd_min, color="darkred", linestyle="--", alpha=0.25)
ax1.set_xlabel("Grain size, Di ($\mu$m)")
ax1.set_ylabel("% percent finer")
ax1.set_title("Grain Size Distribution - Summer")
ax1.grid(True)
ax1.legend()
ax1.set_xscale('log')
ax1.set_xlim((0, 1000))
# Plotting SPRING
#colormap = cm.get_cmap('Blues', spring_st_gsd.shape[1])
#colors = colormap(np.linspace(0, 1, spring_st_gsd.shape[1]))
ax2.set_prop_cycle(color=colors_spring)
ax2.plot(grain_sizes, spring_st_gsd)
ax2.plot(grain_sizes, spring_wc_gsd_max, color="darkred", linestyle="--", alpha=0.25)
ax2.plot(grain_sizes, spring_wc_gsd_avg, color="darkred", label="Water column")
ax2.plot(grain_sizes, spring_wc_gsd_min, color="darkred", linestyle="--", alpha=0.25)
ax2.set_xlabel("Grain size, Di ($\mu$m)")
ax2.set_ylabel("% percent finer")
ax2.set_title("Grain Size Distribution - Spring")
ax2.grid(True)
ax2.legend()
ax2.set_xscale('log')
ax2.set_xlim((0, 1000))
plt.tight_layout()
plt.show()