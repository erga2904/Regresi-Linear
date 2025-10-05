# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)
os.makedirs("static/plots", exist_ok=True)

# Context processor untuk cache-busting
@app.context_processor
def inject_now():
    return {'now': time.time}

# === Muat dataset dan model ===
df = pd.read_csv("Activity.csv")
df_clean = df[(df['Total_Distance'] > 0) & (df['Calories_Burned'] > 1500)].copy()

# Latih & muat model
if not os.path.exists("model.pkl"):
    print("Model tidak ditemukan. Melatih ulang...")
    exec(open("train_model.py").read())
model = joblib.load('model.pkl')

# === Buat plot dasar (sekali saat startup) ===
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
    return render_template('index.html', prediction=None, show_custom_plot=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        distance_km = request.form.get('distance_km', '').strip()
        distance_m = request.form.get('distance_m', '').strip()

        if not distance_km and not distance_m:
            raise ValueError("Silakan isi jarak dalam km atau meter.")

        if distance_m:
            distance_m = float(distance_m)
            if distance_m < 0:
                raise ValueError("Jarak tidak boleh negatif")
            distance_km_value = distance_m / 1000.0
            input_display = f"{distance_m} meter ({distance_km_value:.2f} km)"
        else:
            distance_km_value = float(distance_km)
            if distance_km_value < 0:
                raise ValueError("Jarak tidak boleh negatif")
            input_display = f"{distance_km_value} km"

        pred = model.predict([[distance_km_value]])[0]
        prediction_value = round(pred, 2)

        # Buat scatter plot dengan titik prediksi
        plt.figure(figsize=(6, 4))
        plt.scatter(df_clean['Total_Distance'], df_clean['Calories_Burned'], 
                    alpha=0.5, color='steelblue', label='Data Asli')
        plt.scatter(distance_km_value, prediction_value, 
                    color='red', s=100, label=f'Prediksi Anda: {prediction_value} kalori', zorder=5)
        plt.title('Jarak vs Kalori Terbakar (Termasuk Prediksi Anda)')
        plt.xlabel('Total Jarak (km)')
        plt.ylabel('Kalori Terbakar')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = 'static/plots/scatter_with_prediction.png'
        plt.savefig(plot_path)
        plt.close()

        return render_template('index.html', 
                               prediction=prediction_value, 
                               input_display=input_display,
                               show_custom_plot=True)
    except Exception as e:
        return render_template('index.html', 
                               prediction=f"Error: {str(e)}", 
                               input_display=None,
                               show_custom_plot=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
