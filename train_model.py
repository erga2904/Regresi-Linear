# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Buat folder plots jika belum ada
os.makedirs("static/plots", exist_ok=True)

# Muat dataset
df = pd.read_csv("Activity.csv")

# Pilih kolom
X = df[['Total_Distance']]
y = df['Calories_Burned']

# Filter data realistis: jarak > 0 dan kalori > 1500
df_clean = df[(df['Total_Distance'] > 0) & (df['Calories_Burned'] > 1500)].copy()
X = df_clean[['Total_Distance']]
y = df_clean['Calories_Burned']

# Latih model
model = LinearRegression()
model.fit(X, y)

# Simpan model
joblib.dump(model, 'model.pkl')
print("✅ Model disimpan sebagai 'model.pkl'")
