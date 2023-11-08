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
    labels_runs = ["T1-T2","T2-T3","T3-T4"]
    sensors = ["T2","T3","T4"]
    #
    fig1,ax1 = plt.subplots(2)
    # fig2,ax2 = plt.subplots()
    for i,run in enumerate(runs):
        df_ke = pd.read_pickle(f"./{probe}/PEdata/{run}_ke.pkz", compression="zip")
        df_vel = pd.read_pickle(f"./{probe}/PEdata/{run}_Velocity.pkz", compression="zip")
        df_ke["Time"] = pd.to_datetime(df_ke["Time"],unit="s")
        df_vel["Time"] = pd.to_datetime(df_vel["Time"],unit="s")
        ax1[0].plot(df_ke["Time"].to_numpy(), df_ke[sensors[i]].to_numpy(), label=labels_runs[i])
        ax1[1].plot(df_vel["Time"].to_numpy(), df_vel[sensors[i]].to_numpy(), label=labels_runs[i])
        # df_ke["Time"] = df_ke["time"].to_datetime()
        df_ke.to_csv(f"./{run}_ke.csv", index=False)
        df_vel.to_csv(f"./{run}_vel.csv", index=False)
        #
    # ax1.set_title("Ke")
    ax1[0].grid()
    ax1[0].legend()
    ax1[1].grid()
    ax1[1].legend()
    # ax2.set_title("Flux")
    # ax2.grid()
    # ax2.legend()

    plt.show()

    return


if __name__ == "__main__":
    main()
    print(f"End")
    sys.exit(0)