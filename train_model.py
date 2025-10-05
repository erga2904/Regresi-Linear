# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 1. Muat dataset
df = pd.read_csv("Activity.csv")

# 2. Pilih kolom yang relevan
X = df[['Total_Distance']]  # Input: jarak (km)
y = df['Calories_Burned']   # Output: kalori terbakar

# 3. Hapus baris dengan nilai 0 di kedua kolom (opsional, untuk data lebih realistis)
df = df[(df['Total_Distance'] > 0) & (df['Calories_Burned'] > 0)]
X = df[['Total_Distance']]
y = df['Calories_Burned']

# 4. Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Simpan model ke file
joblib.dump(model, 'model.pkl')
print("Model berhasil disimpan sebagai 'model.pkl'")

# Opsional: Tampilkan koefisien
print(f"Intercept: {model.intercept_:.2f}")
print(f"Koefisien (Kalori per km): {model.coef_[0]:.2f}")