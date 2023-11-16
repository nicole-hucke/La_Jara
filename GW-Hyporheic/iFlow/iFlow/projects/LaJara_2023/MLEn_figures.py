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
    probe = "T8"
    runs = [3,4,5]
    labels_runs = ["30","45","60"]
    #
    fig1,ax1 = plt.subplots(2)
    # fig2,ax2 = plt.subplots()
    df_all_vel = pd.DataFrame()
    df_all_dif = pd.DataFrame()
    df_all_lims = pd.DataFrame()
    for i,run in enumerate(runs):
        df_dif = pd.read_pickle(f"./{probe}/PEdata/{run}_Diffusivity.pkz", compression="zip")
        df_vel = pd.read_pickle(f"./{probe}/PEdata/{run}_Velocity.pkz", compression="zip")
        print(df_dif.columns)
        print(df_vel.columns)
        df_dif["Time"] = pd.to_datetime(df_dif["Time"],unit="s")
        df_vel["Time"] = pd.to_datetime(df_vel["Time"],unit="s")
        ax1[0].plot(df_dif["Time"].to_numpy(), df_dif["Dif"].to_numpy(), label=labels_runs[i])
        ax1[1].plot(df_vel["Time"].to_numpy(), df_vel["Vel"].to_numpy(), label=labels_runs[i])

        df_limits = df_vel[["Time", "VelL", "VelI"]] # these are the error in the estimation of the velocity
        df_vel = df_vel[["Time", "Vel"]] # select only the 'Time' and 'Vel' columns
        df_dif = df_dif[["Time", "Dif"]] # select only the 'Time' and 'Dif' columns

        df_vel.rename(columns={"Vel": labels_runs[i]}, inplace=True) # rename the 'Vel' column to the current run label
        df_dif.rename(columns={"Dif": labels_runs[i]}, inplace=True) # rename the 'Dif' column to the current run label
        df_limits.rename(columns={"VelL": labels_runs[i]+"L", "VelI": labels_runs[i]+"I"}, inplace=True) # rename the 'Vel' column to the current run label
        # If this is the first run, copy the data to df_all_vel
        if df_all_vel.empty and df_all_dif.empty:
            df_all_vel = df_vel
            df_all_dif = df_dif
            df_all_lims = df_limits
        else:
        # If this is not the first run, merge the data with df_all_vel
            df_all_vel = pd.merge(df_all_vel, df_vel, on="Time", how="outer")
            df_all_dif = pd.merge(df_all_dif, df_dif, on="Time", how="outer")
            df_all_lims = pd.merge(df_all_lims, df_limits, on="Time", how="outer")
        #   
    ax1[0].grid()
    ax1[0].legend()
    ax1[1].grid()
    ax1[1].legend()
    df_all_vel.to_csv(f"./{probe}_fluxes_MLEn.csv", index=False)
    df_all_dif.to_csv(f"./{probe}_diffusivity_MLEn.csv", index=False)
    df_all_lims.to_csv(f"./{probe}_limits_MLEn.csv", index=False)
    plt.show()
    return


if __name__ == "__main__":
    main()
    print(f"End")
    sys.exit(0)