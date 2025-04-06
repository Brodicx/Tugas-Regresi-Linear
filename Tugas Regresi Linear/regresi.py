# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load Dataset
df = pd.read_csv("C:\Users\HD\Documents\Ngampus\Semester 4\Otomata\Tugas Regresi Linear\housing.csv")
print(df.head())

# Cek nilai yang hilang
print(df.isnull().sum())

# One-Hot Encoding untuk variabel kategori (Lokasi)
df = pd.get_dummies(df, columns=['Lokasi'], drop_first=True)

# Pilih Fitur (X) dan Target (y)
X = df[['LuasTanah'] + list(df.columns[df.columns.str.startswith('Lokasi_')])]  # Tambahkan fitur lokasi
y = df['Harga']  # Target dependen

# Pisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat Model Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi Data Uji
y_pred = model.predict(X_test)

# Evaluasi Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Visualisasi Hasil Prediksi untuk Luas Tanah (hanya visualisasi 2D)
plt.scatter(X_test['LuasTanah'], y_test, color='blue', label='Data Aktual')
plt.scatter(X_test['LuasTanah'], y_pred, color='red', label='Prediksi', alpha=0.5)
plt.xlabel('Luas Tanah')
plt.ylabel('Harga Rumah')
plt.legend()
plt.show()
