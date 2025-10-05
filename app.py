# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
os.makedirs("static/plots", exist_ok=True)

# Jika model belum ada, latih ulang
if not os.path.exists("model.pkl"):
    print("Model tidak ditemukan. Melatih model...")
    exec(open("train_model.py").read())

# Muat model
model = joblib.load('model.pkl')

# Muat dataset untuk visualisasi
df = pd.read_csv("Activity.csv")
df_clean = df[(df['Total_Distance'] > 0) & (df['Calories_Burned'] > 1500)].copy()

# === Buat plot (jika belum ada) ===
if not os.path.exists("static/plots/scatter_data.png"):
    plt.figure(figsize=(6, 4))
    plt.scatter(df_clean['Total_Distance'], df_clean['Calories_Burned'], alpha=0.6, color='steelblue')
    plt.title('Jarak vs Kalori Terbakar (Data Asli)')
    plt.xlabel('Total Jarak (km)')
    plt.ylabel('Kalori Terbakar')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/scatter_data.png')
    plt.close()

if not os.path.exists("static/plots/actual_vs_pred.png"):
    from sklearn.model_selection import train_test_split
    X = df_clean[['Total_Distance']]
    y = df_clean['Calories_Burned']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Kalori Aktual vs Prediksi')
    plt.xlabel('Kalori Aktual')
    plt.ylabel('Kalori Prediksi')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/actual_vs_pred.png')
    plt.close()

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        distance = float(request.form['distance'])
        if distance < 0:
            raise ValueError("Jarak tidak boleh negatif")
        pred = model.predict([[distance]])[0]
        return render_template('index.html', prediction=round(pred, 2), input_distance=distance)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", input_distance=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
