import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from tqdm import tqdm

class ProgressFuncAnimation(animation.FuncAnimation):
    def __init__(self, *args, **kwargs):
        self.pbar = tqdm(total=kwargs.get('frames'))
        animation.FuncAnimation.__init__(self, *args, **kwargs)

    def _draw_frame(self, framedata):
        self.pbar.update(1)
        animation.FuncAnimation._draw_frame(self, framedata)

    def _stop(self, *args):
        self.pbar.close()
        animation.FuncAnimation._stop(self, *args)

# Importing all the GW topography data
directory = 'C:/Users/huck4481/Documents/GitHub/La_Jara/Diver_Data/WSE/XS_topo'
# Dictionary to store the data frames
topo_dict = {}
# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Read the CSV file into a data frame
        df = pd.read_csv(file_path)
        # Remove the '.csv' from the filename to use as the dictionary key
        key = filename[:-4]
        # Store the data frame in the dictionary
        topo_dict[key] = df 

################################# GW1 #################################
        
# Define the well properties
well_location = 350  
well_top_elevation = 2722.372  
well_length = 1.20  

# Import the WSE data
directory = 'C:/Users/huck4481/Documents/GitHub/La_Jara/Diver_Data/WSE/SM21/GW1_SM21_WSE.csv'
water_table_data = pd.read_csv(directory, index_col='Date_Time', parse_dates=True)

# Calculate the bottom elevation of the well
well_bottom_elevation = well_top_elevation - well_length

#### Animate the water table elevation ####

fig, ax = plt.subplots(figsize=(10, 5))

# Plot the topography and the well
ax.plot(topo_dict['GW1_topo']['distance'], topo_dict['GW1_topo']['elevation'], color='black')
ax.vlines(well_location, well_bottom_elevation, well_top_elevation, colors='r', linewidth=3)
ax.text(well_location, well_top_elevation, 'GW1', ha='center', va='bottom')

# Add a line for the water table. Its y-data will be updated in the animate function.
water_table_line, = ax.plot(topo_dict['GW1_topo']['distance'], [0]*len(topo_dict['GW1_topo']['distance']), 'b--')

# Set the labels and title
ax.set_xlabel('Lateral Distance')
ax.set_ylabel('Elevation')
ax.set_title('Cross Sectional Topography of GW1')
ax.grid(True)

# Set the vertical limits to the topography elevations
ax.set_ylim(well_bottom_elevation, topo_dict['GW1_topo']['elevation'].max())

# Precompute the y-data for all frames, using every second data point
y_data = [[water_table_data['WSE'].iloc[i]]*len(topo_dict['GW1_topo']['distance']) for i in range(0, len(water_table_data), 5)]

# This function will be called for each frame of the animation.
def animate(i):
    # Update the y-data of the water table line
    water_table_line.set_ydata(y_data[i])
    # Update the title with the current date
    # Note: we multiply i by 2 to get the correct date because we're using every second data point
    ax.set_title(f'Cross Sectional Topography of GW1 - {water_table_data.index[i*5].strftime("%m/%d/%Y")}')

# Create the animation with a progress bar and a smaller interval
# Note: we divide the number of frames by 2 because we're using every second data point
ani = ProgressFuncAnimation(fig, animate, frames=len(water_table_data)//5, interval=1)

# Save the animation as a GIF
ani.save('GW1_SM21.gif', writer='pillow')

plt.show()