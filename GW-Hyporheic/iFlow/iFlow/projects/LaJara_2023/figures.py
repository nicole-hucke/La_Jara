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
    runs = [0,1,2]
    labels_runs = ["30","45","60"]
    sensors = ["T2","T3","T4"]
    #
    fig1,ax1 = plt.subplots(2)
    # fig2,ax2 = plt.subplots()
    df_all_vel = pd.DataFrame()
    df_all_ke = pd.DataFrame()
    for i,run in enumerate(runs):
        df_ke = pd.read_pickle(f"./{probe}/PEdata/{run}_ke.pkz", compression="zip")
        df_vel = pd.read_pickle(f"./{probe}/PEdata/{run}_Velocity.pkz", compression="zip")
        df_ke["Time"] = pd.to_datetime(df_ke["Time"],unit="s")
        df_vel["Time"] = pd.to_datetime(df_vel["Time"],unit="s")
        ax1[0].plot(df_ke["Time"].to_numpy(), df_ke[sensors[i]].to_numpy(), label=labels_runs[i])
        ax1[1].plot(df_vel["Time"].to_numpy(), df_vel[sensors[i]].to_numpy(), label=labels_runs[i])
        # df_ke["Time"] = df_ke["time"].to_datetime()
        #df_flux.rename(columns={sensors[i]: labels_runs[i]}, inplace=True)
        #df_all_vel = pd.merge(df_all_vel, df_flux, on="Time", how="outer")
        df_vel = df_vel[["Time", sensors[i]]] # select only the 'Time' and the current sensor columns
        df_ke = df_ke[["Time", sensors[i]]] 
        df_vel.rename(columns={sensors[i]: labels_runs[i]}, inplace=True) # rename the sensor column to the current run label
        df_ke.rename(columns={sensors[i]: labels_runs[i]}, inplace=True)
        # If this is the first run, copy the data to df_all_vel
        if df_all_vel.empty and df_all_ke.empty:
            df_all_vel = df_vel
            df_all_ke = df_ke
        else:
            # If this is not the first run, merge the data with df_all_vel
            df_all_vel = pd.merge(df_all_vel, df_vel, on="Time", how="outer")
            df_all_ke = pd.merge(df_all_ke, df_ke, on="Time", how="outer")
        #
    # ax1.set_title("Ke")
    ax1[0].grid()
    ax1[0].legend()
    ax1[1].grid()
    ax1[1].legend()
    # ax2.set_title("Flux")
    # ax2.grid()
    # ax2.legend()
    df_all_vel.to_csv(f"./{probe}_fluxes.csv", index=False)
    df_all_ke.to_csv(f"./{probe}_K.csv", index=False)
    plt.show()
    return


if __name__ == "__main__":
    main()
    print(f"End")
    sys.exit(0)