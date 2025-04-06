import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
df = pd.read_csv("housing.csv")

# 2. Konversi kolom lokasi menjadi numerik
le = LabelEncoder()
df["Lokasi"] = le.fit_transform(df["Lokasi"])

# 3. Fitur dan Label
X = df[["LuasTanah", "Lokasi"]]
y = df["Harga"]

# 4. Bagi Data (Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Buat Model Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Simpan Model ke `model.pkl`
joblib.dump((model, le), "model.pkl")

print("✅ Model berhasil disimpan sebagai model.pkl!")
