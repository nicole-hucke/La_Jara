"""
Description
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylustrator
"""
Main function
"""
pylustrator.start()

def main():
    probe = "T1"
    run = 6
    labels_runs = ["30","45","60"]
    sensors = ["T2","T3","T4"]
    df_ke = pd.read_pickle(f"./{probe}/PEdata/{run}_ke.pkz", compression="zip")
    df_vel = pd.read_pickle(f"./{probe}/PEdata/{run}_Velocity.pkz", compression="zip")
    df_ke["Time"] = pd.to_datetime(df_ke["Time"],unit="s")
    df_vel["Time"] = pd.to_datetime(df_vel["Time"],unit="s")

    fig1,ax1 = plt.subplots(2)
    for i, sensor in enumerate(sensors):
        ax1[0].plot(df_ke["Time"].to_numpy(), df_ke[sensor].to_numpy(), label=labels_runs[i]+'cm')
        ax1[1].plot(df_vel["Time"].to_numpy(), df_vel[sensor].to_numpy(), label=labels_runs[i]+'cm')
        df_vel.rename(columns={sensor: labels_runs[i]}, inplace=True) # rename the sensor column to the current run label
        df_ke.rename(columns={sensor: labels_runs[i]}, inplace=True) # rename the sensor column to the current run label

    ax1[0].grid()
    ax1[0].legend()
    ax1[0].title.set_text(f"{probe} Ke")
    ax1[1].grid()
    ax1[1].legend()
    ax1[1].title.set_text(f"{probe} Flux (cm/s)")
    df_vel.to_csv(f"./{probe}_fluxes_analytical.csv", index=False)
    df_ke.to_csv(f"./{probe}_K_analytical.csv", index=False)
    plt.show()
    return


if __name__ == "__main__":
    main()
    print(f"End")
    sys.exit(0)