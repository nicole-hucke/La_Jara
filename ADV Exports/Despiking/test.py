import pandas as pd

csv_file_path = 'T1A_SP23_ADV.csv'
data = pd.read_csv(csv_file_path)
print(data.columns)