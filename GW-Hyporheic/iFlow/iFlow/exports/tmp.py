import pandas as pd

df = pd.read_csv("Probe16AWT.csv")
df["Time"] = pd.to_datetime(df["Time"], unit = "s")

df.to_csv("Probe16AWT.csv",index=False)