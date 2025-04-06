import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("housing.csv")

# One-Hot Encoding untuk Lokasi
df = pd.get_dummies(df, columns=['Lokasi'], drop_first=True)

# Pilih fitur
X = df[['LuasTanah'] + list(df.columns[df.columns.str.startswith('Lokasi_')])]
y = df['Harga']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model dan latih
model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(luas_tanah, lokasi):
    lokasi_features = {col: 0 for col in df.columns if col.startswith("Lokasi_")}
    if f'Lokasi_{lokasi}' in lokasi_features:
        lokasi_features[f'Lokasi_{lokasi}'] = 1

    input_data = np.array([[luas_tanah] + list(lokasi_features.values())])
    return model.predict(input_data)[0]
