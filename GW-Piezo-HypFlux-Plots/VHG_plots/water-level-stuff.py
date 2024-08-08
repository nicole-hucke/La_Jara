########################
#Water-level-stuff

# Importing libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Checking environment and working directory
import os
os.getcwd()
working_directory = "C:/Users/bern9483/Documents/Nicole/water-level-stuff/GW-Piezo-HypFlux-Plots/"
os.chdir(working_directory)
results_directory = "./plots/"

###################################################################### Calculating VHG - GW
# Filenames and directories
filename1 = "./GW_WSE/piezometer_WSE_2023_corrected.csv"
filename2 = "./GW_WSE/SM23_GWSE_corrected.csv"
filename3 = "./GW_WSE/SP23_GWSE_corrected.csv"

# Reading files
dataframe1 = pd.read_csv(filename1)
dataframe2 = pd.read_csv(filename2)
dataframe3 = pd.read_csv(filename3)
dataframe2.head()

# Setting index
dataframe1.set_index("Date_Time", inplace=True)
dataframe2.set_index("Date_Time", inplace=True)
dataframe3.set_index("Date_Time", inplace=True)

# Calculation for the summer
tempCalc1 = dataframe1[["P2B", "P5B"]].dropna()
tempCalc2 = dataframe2[["GW3", "GW8"]].dropna()
VHG_GW_summer = tempCalc1.join(tempCalc2, how="outer")
tempCalc2.join(tempCalc1, how="outer")
VHG_GW_summer["VHG_GW_downstream"] = VHG_GW_summer["GW3"] - VHG_GW_summer["P2B"]
VHG_GW_summer["VHG_GW_upstream1"] = VHG_GW_summer["GW8"] - VHG_GW_summer["P5B"]
VHG_GW_summer["VHG_GW_upstream2"] = VHG_GW_summer["GW9"] - VHG_GW_summer["P5B"]

###################################################################### Calculating VHG - PZ
# Filenames and directories
filename1 = "./Piezo_VHG/VHG_B-A_2023.csv"
filename2 = "./Piezo_VHG/VHG_C-B_2023.csv"

# Reading files
dataframe1 = pd.read_csv(filename1)
dataframe2 = pd.read_csv(filename2)
dataframe1.head()
dataframe2.head()

# Setting index
dataframe1.set_index("Date_Time", inplace=True)
dataframe2.set_index("Date_Time", inplace=True)

# Calculation for the summer
tempCalc1 = dataframe1[["P2", "P5"]].dropna()
tempCalc1.columns = ["downstream_bottom", "upstream_bottom"]
tempCalc2 = dataframe2[["P2", "P5"]].dropna()
tempCalc2.columns = ["downstream_top", "upstream_top"]
VHG_PZ_summer = tempCalc1.join(tempCalc2, how="outer")

###################################################################### Calculating HYP FLUX
# Filenames and directories
filename1 = "./Hyp_Flux/hyporheic_fluxes_SM23_top_averagedmethod.csv"
filename2 = "./Hyp_Flux/hyporheic_fluxes_SP23_top_averagedmethod.csv"

# Reading files
dataframe1 = pd.read_csv(filename1)
dataframe2 = pd.read_csv(filename2)
dataframe1.head()
dataframe2.head()

# Setting index
dataframe1.set_index("DateTime", inplace=True)
dataframe2.set_index("DateTime", inplace=True)

# Calculation for the summer
HYP_flux_summer = dataframe1[["T2", "T7"]].dropna()
HYP_flux_summer.columns = ["downstream", "upstream"]

###################################################################### Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
# Plotting VHG-GW upstream
axs[0,0].plot(VHG_GW_summer.index, VHG_GW_summer["VHG_GW_upstream1"], color="orchid")
#axs[0,0].plot(VHG_GW_summer.index, VHG_GW_summer["VHG_GW_upstream2"], color="thistle")
axs[0,0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axs[0,0].set_xlabel("Date")
axs[0,0].set_ylabel("VHG")
axs[0,0].set_title("Groundwater")
# Plotting VHG-PZ upstream
#axs[1,0].plot(VHG_PZ_summer.index, VHG_PZ_summer["upstream_top"], color="yellowgreen")
axs[1,0].plot(VHG_PZ_summer.index, VHG_PZ_summer["upstream_bottom"], color="springgreen")
axs[1,0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axs[1,0].set_xlabel("Date")
axs[1,0].set_ylabel("VHG")
axs[1,0].set_title("Piezometer")
# Plotting HYP-flux upstream
axs[2,0].plot(HYP_flux_summer.index, HYP_flux_summer["upstream"], color="orange")
axs[2,0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axs[2,0].set_xlabel("Date")
axs[2,0].set_ylabel("FLUX")
axs[2,0].set_title("Hyp. Probe")
# Plotting VHG-GW downstream
axs[0,1].plot(VHG_GW_summer.index, VHG_GW_summer["VHG_GW_downstream"], color="orchid")
axs[0,1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axs[0,1].set_xlabel("Date")
axs[0,1].set_ylabel("VHG")
axs[0,1].set_title("Groundwater")
# Plotting VHG-PZ downstream
#axs[1,1].plot(VHG_PZ_summer.index, VHG_PZ_summer["downstream_top"], color="yellowgreen")
axs[1,1].plot(VHG_PZ_summer.index, VHG_PZ_summer["downstream_bottom"], color="springgreen")
axs[1,1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axs[1,1].set_xlabel("Date")
axs[1,1].set_ylabel("VHG")
axs[1,1].set_title("Piezometer")
# Plotting HYP-flux downstream
axs[2,1].plot(HYP_flux_summer.index, HYP_flux_summer["downstream"], color="orange")
axs[2,1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axs[2,1].set_xlabel("Date")
axs[2,1].set_ylabel("FLUX")
axs[2,1].set_title("Hyp. Probe")
#plt.grid(True)
axs[0, 0].xaxis.set_major_locator(mdates.DayLocator(interval=500))
axs[0, 1].xaxis.set_major_locator(mdates.DayLocator(interval=500))
axs[1, 0].xaxis.set_major_locator(mdates.DayLocator(interval=800))
axs[1, 1].xaxis.set_major_locator(mdates.DayLocator(interval=800))
axs[2, 0].xaxis.set_major_locator(mdates.DayLocator(interval=800))
axs[2, 1].xaxis.set_major_locator(mdates.DayLocator(interval=800))
plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[2, 0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[2, 1].xaxis.get_majorticklabels(), rotation=45)

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%Y"))
plt.tight_layout()  # Adjust the layout to fit labels
plt.show()


