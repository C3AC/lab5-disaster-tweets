"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 3: Frecuencia de palabras (unigramas) y n-gramas (bigramas/trigramas)
por clase (desastre real vs no desastre)
"""
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("../data/train_clean.csv")
df = df.dropna(subset=['text_clean'])
df = df[df['text_clean'].str.len() > 0]

desastre = df[df['target'] == 1]['text_clean']
no_desastre = df[df['target'] == 0]['text_clean']


def top_ngrams(corpus, ngram_range=(1, 1), top_n=20):
    vec = CountVectorizer(ngram_range=ngram_range)
    matrix = vec.fit_transform(corpus)
    sums = matrix.sum(axis=0)
    freqs = [(word, sums[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
    return freqs[:top_n]


print("=" * 60)
print("TOP 20 UNIGRAMAS - Tweets de DESASTRE REAL (target=1)")
print("=" * 60)
uni_desastre = top_ngrams(desastre, (1, 1), 20)
for w, c in uni_desastre:
    print(f"{w:20s} {c}")

print("\n" + "=" * 60)
print("TOP 20 UNIGRAMAS - Tweets que NO son desastre (target=0)")
print("=" * 60)
uni_no_desastre = top_ngrams(no_desastre, (1, 1), 20)
for w, c in uni_no_desastre:
    print(f"{w:20s} {c}")

print("\n" + "=" * 60)
print("TOP 15 BIGRAMAS - Tweets de DESASTRE REAL (target=1)")
print("=" * 60)
bi_desastre = top_ngrams(desastre, (2, 2), 15)
for w, c in bi_desastre:
    print(f"{w:30s} {c}")

print("\n" + "=" * 60)
print("TOP 15 BIGRAMAS - Tweets que NO son desastre (target=0)")
print("=" * 60)
bi_no_desastre = top_ngrams(no_desastre, (2, 2), 15)
for w, c in bi_no_desastre:
    print(f"{w:30s} {c}")

print("\n" + "=" * 60)
print("TOP 10 TRIGRAMAS - Tweets de DESASTRE REAL (target=1)")
print("=" * 60)
tri_desastre = top_ngrams(desastre, (3, 3), 10)
for w, c in tri_desastre:
    print(f"{w:40s} {c}")

# Palabras que aparecen en ambas categorias (top 50 de cada una)
set_desastre = set(w for w, c in top_ngrams(desastre, (1, 1), 50))
set_no_desastre = set(w for w, c in top_ngrams(no_desastre, (1, 1), 50))
comunes = set_desastre & set_no_desastre
print("\n" + "=" * 60)
print(f"Palabras en el TOP 50 de AMBAS categorias ({len(comunes)}):")
print("=" * 60)
print(sorted(comunes))

# Guardar resultados a CSV para el informe
pd.DataFrame(uni_desastre, columns=['palabra', 'frecuencia']).to_csv(
    "../data/unigramas_desastre.csv", index=False)
pd.DataFrame(uni_no_desastre, columns=['palabra', 'frecuencia']).to_csv(
    "../data/unigramas_no_desastre.csv", index=False)
pd.DataFrame(bi_desastre, columns=['bigrama', 'frecuencia']).to_csv(
    "../data/bigramas_desastre.csv", index=False)
pd.DataFrame(bi_no_desastre, columns=['bigrama', 'frecuencia']).to_csv(
    "../data/bigramas_no_desastre.csv", index=False)

print("\nCSVs de frecuencias guardados en ../data/")
