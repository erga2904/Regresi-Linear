import pandas as pd

# Ganti dengan nama file datasetmu
df = pd.read_csv("activity.csv")

# Tampilkan 5 data pertama
print("5 Baris Pertama:")
print(df.head())

# Tampilkan nama kolom
print("\nNama Kolom:")
print(df.columns.tolist())

# Cek info dasar (jumlah data, tipe data, missing value)
print("\nInfo Dataset:")
print(df.info())