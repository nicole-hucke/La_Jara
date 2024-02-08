"""
Description
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylustrator
from matplotlib.ticker import FormatStrFormatter
"""
Main function
"""
pylustrator.start()

def main():
    probe = "T8"
    run = 7
    df_dif = pd.read_pickle(f"./{probe}/PEdata/{run}_Diffusivity.pkz", compression="zip")
    df_vel = pd.read_pickle(f"./{probe}/PEdata/{run}_Velocity.pkz", compression="zip")
    df_dif["Time"] = pd.to_datetime(df_dif["Time"],unit="s")
    df_vel["Time"] = pd.to_datetime(df_vel["Time"],unit="s")
    fig1,ax1 = plt.subplots(2)
    ax1[0].plot(df_dif['Time'], df_dif['Dif'].to_numpy(), label='60 cm')
    ax1[1].plot(df_vel['Time'], df_vel['Vel'].to_numpy(), label='60 cm')
    ax1[0].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ax1[1].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    ax1[0].grid()
    ax1[0].legend()
    ax1[0].title.set_text(f"{probe} Ke")
    ax1[1].grid()
    ax1[1].legend()
    ax1[1].title.set_text(f"{probe} Flux (cm/s)")
    df_vel.to_csv(f"./{probe}_fluxes_MLEn_avg.csv", index=False)
    df_dif.to_csv(f"./{probe}_dif_MLEn_avg.csv", index=False)
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).axes[0].set(position=[0.125, 0.5773, 0.775, 0.35])
    #% end: automatic generated code from pylustrator
    plt.show()
    return


if __name__ == "__main__":
    main()
    print(f"End")
    sys.exit(0)