import os
import numpy as np
import joblib
from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Inisialisasi Flask
app = Flask(__name__)

# Load dataset untuk mendapatkan daftar lokasi
df = pd.read_csv("housing.csv")
lokasi_list = sorted(df['Lokasi'].unique().tolist())

# Load model dan encoder lokasi
if os.path.exists("model.pkl"):
    model, le = joblib.load("model.pkl")
else:
    raise FileNotFoundError("File model.pkl tidak ditemukan. Jalankan train_model.py terlebih dahulu.")

# Inisialisasi dan fit encoder
le = LabelEncoder()
le.fit(lokasi_list)

# Simpan encoder dan model
joblib.dump((model, le), "model.pkl")

# Fungsi untuk membuat grafik prediksi
def generate_graph(luas_tanah, harga_prediksi):
    luas_sample = np.linspace(50, 500, 100)  # Contoh luas tanah
    noise = np.random.normal(0, 10, luas_sample.shape)  # Tambahkan noise
    harga_sample = model.predict(np.column_stack([luas_sample, np.zeros_like(luas_sample)])) + noise  # Asumsi lokasi = 0

    plt.figure(figsize=(6, 4))
    plt.plot(luas_sample, harga_sample, label="Regresi Linear", color="blue")
    plt.scatter(luas_tanah, harga_prediksi, color="red", marker="o", label="Prediksi Anda")
    plt.xlabel("Luas Tanah (m²)")
    plt.ylabel("Harga Rumah")
    plt.legend()
    plt.grid()

    # Simpan ke buffer
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{graph_url}"

# Route utama
@app.route("/", methods=["GET", "POST"])
def index():
    graph_url = None
    prediction_text = None

    if request.method == "POST":
        try:
            luas_tanah = float(request.form["luas_tanah"])
            lokasi = request.form["lokasi"]

            # Ubah lokasi ke bentuk numerik
            lokasi_encoded = le.transform([lokasi])[0]

            # Prediksi harga
            input_data = pd.DataFrame([[luas_tanah, lokasi_encoded]], columns=['LuasTanah', 'Lokasi'])
            harga_prediksi = model.predict(input_data)[0]

            # Buat grafik
            graph_url = generate_graph(luas_tanah, harga_prediksi)
            prediction_text = f"Prediksi harga rumah: Rp {harga_prediksi:,.2f}"

        except Exception as e:
            prediction_text = f"Terjadi kesalahan: {str(e)}"

    return render_template("index.html", 
                         prediction_text=prediction_text, 
                         graph_url=graph_url,
                         lokasi_list=lokasi_list)

# Jalankan Aplikasi
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
