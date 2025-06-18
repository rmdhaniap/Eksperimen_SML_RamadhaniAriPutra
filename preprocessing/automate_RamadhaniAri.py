import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Memuat dataset
df = pd.read_csv('diabetes_raw.csv')

# Pisahkan fitur numerik dan kategorikal
num_features = df.select_dtypes(include=[np.number]).columns
cat_features = df.select_dtypes(include=['object']).columns

# Tangani outlier menggunakan metode IQR
Q1 = df[num_features].quantile(0.25)
Q3 = df[num_features].quantile(0.75)
IQR = Q3 - Q1

# Filter baris tanpa outlier
filter_outliers = ~((df[num_features] < (Q1 - 1.5 * IQR)) |
                    (df[num_features] > (Q3 + 1.5 * IQR))).any(axis=1)

# Terapkan filter ke seluruh dataframe
df = df[filter_outliers]

# Standarisasi fitur numerik
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Encode fitur kategorikal
label_encoder = LabelEncoder()
for col in cat_features:
    df[col] = label_encoder.fit_transform(df[col])

# Pastikan folder 'preprocessing' tersedia
os.makedirs('preprocessing', exist_ok=True)

# Simpan hasil preprocessing ke folder
df.to_csv('preprocessing/diabetes_data_preprocessing.csv', index=False)

print("✅ Preprocessing selesai. File disimpan di 'preprocessing/obesity_data_preprocessing.csv'")
