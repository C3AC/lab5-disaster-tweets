"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 1: Carga de datos y exploracion inicial
"""
import pandas as pd

# ---- Carga del dataset ----
df = pd.read_csv("../data/train.csv")

print("Dimensiones del dataset:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)

print("\nPrimeras filas:")
print(df.head())

print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nPorcentaje de nulos:")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\nDistribucion de la variable objetivo (target):")
print(df['target'].value_counts())
print((df['target'].value_counts(normalize=True) * 100).round(2))

print("\nNumero de keywords distintos:", df['keyword'].nunique())
print("Numero de locations distintas:", df['location'].nunique())

print("\nLongitud de texto (caracteres):")
df['text_len'] = df['text'].str.len()
print(df['text_len'].describe())

print("\nLongitud de texto (palabras):")
df['word_count'] = df['text'].str.split().str.len()
print(df['word_count'].describe())

# Duplicados
print("\nFilas duplicadas (por texto):", df.duplicated(subset=['text']).sum())

df.to_csv("../data/train_explored.csv", index=False)
