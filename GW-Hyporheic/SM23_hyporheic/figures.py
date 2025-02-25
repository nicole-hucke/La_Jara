"""
Description
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylustrator
import os

pylustrator.start()

def main():
    # read the csv file of the probe I want to plot
    probe = "T8"
    path = os.path.join("flux", f"{probe}_flux_SM23.csv")
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"])
    depths = ["30","45","60"]

    df = df[df["Time"] >= "2023-06-22"]  # exclude data before a certain date

    # plotting the data: 
    plt.figure(figsize=(10, 5), dpi=300)
    for depth in depths:
        plt.plot(df['Time'].to_numpy(), df[depth].to_numpy(), label=f"{depth} cm")

    plt.title(f"Flux at {probe} probe, Summer 2023")
    plt.xlabel('Date')
    plt.ylabel('Flux (m³/s)')
    plt.xticks(rotation=15)  # rotate x-axis labels by 45 degrees
    plt.legend()

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).axes[0].grid(True)
    plt.figure(1).axes[0].set(position=[0.1179, 0.1431, 0.775, 0.77])
    #% end: automatic generated code from pylustrator

    plt.show()


if __name__ == "__main__":
    main()