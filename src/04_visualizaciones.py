"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 4: Analisis exploratorio visual
- Distribucion de la clase objetivo
- Nube de palabras por categoria
- Histogramas de palabras mas frecuentes por categoria
- Distribucion de longitud de tweets por categoria
"""
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

plt.rcParams['figure.dpi'] = 150

df = pd.read_csv("../data/train_clean.csv")
df = df.dropna(subset=['text_clean'])
df = df[df['text_clean'].str.len() > 0]

desastre = df[df['target'] == 1]['text_clean']
no_desastre = df[df['target'] == 0]['text_clean']

FIG_DIR = "../figures/"

# ---------------------------------------------------------------
# 1. Distribucion de la clase objetivo
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['target'].value_counts().sort_index()
labels = ['No desastre (0)', 'Desastre real (1)']
bars = ax.bar(labels, counts.values, color=['#4C72B0', '#C44E52'])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 20, str(val),
            ha='center', fontweight='bold')
ax.set_title('Distribucion de la variable objetivo (target)')
ax.set_ylabel('Cantidad de tweets')
plt.tight_layout()
plt.savefig(FIG_DIR + "01_distribucion_target.png")
plt.close()

# ---------------------------------------------------------------
# 2. Nube de palabras - Desastre real
# ---------------------------------------------------------------
wc_desastre = WordCloud(width=900, height=500, background_color='white',
                         colormap='Reds', max_words=100).generate(
    ' '.join(desastre))
plt.figure(figsize=(10, 6))
plt.imshow(wc_desastre, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras - Tweets de desastre real (target=1)', fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR + "02_wordcloud_desastre.png")
plt.close()

# ---------------------------------------------------------------
# 3. Nube de palabras - No desastre
# ---------------------------------------------------------------
wc_no_desastre = WordCloud(width=900, height=500, background_color='white',
                            colormap='Blues', max_words=100).generate(
    ' '.join(no_desastre))
plt.figure(figsize=(10, 6))
plt.imshow(wc_no_desastre, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras - Tweets que NO son desastre (target=0)', fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR + "03_wordcloud_no_desastre.png")
plt.close()

# ---------------------------------------------------------------
# 4. Nube de palabras - Dataset completo
# ---------------------------------------------------------------
wc_general = WordCloud(width=900, height=500, background_color='white',
                        colormap='viridis', max_words=100).generate(
    ' '.join(df['text_clean']))
plt.figure(figsize=(10, 6))
plt.imshow(wc_general, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras - Todo el dataset', fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR + "04_wordcloud_general.png")
plt.close()


def get_top_n(corpus, n=15, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range)
    matrix = vec.fit_transform(corpus)
    sums = matrix.sum(axis=0)
    freqs = [(word, sums[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(freqs, key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------
# 5. Histograma top 15 unigramas - Desastre
# ---------------------------------------------------------------
top_d = get_top_n(desastre, 15)
fig, ax = plt.subplots(figsize=(8, 6))
words, counts_ = zip(*top_d)
ax.barh(words[::-1], counts_[::-1], color='#C44E52')
ax.set_title('Top 15 palabras mas frecuentes - Desastre real')
ax.set_xlabel('Frecuencia')
plt.tight_layout()
plt.savefig(FIG_DIR + "05_top_palabras_desastre.png")
plt.close()

# ---------------------------------------------------------------
# 6. Histograma top 15 unigramas - No desastre
# ---------------------------------------------------------------
top_nd = get_top_n(no_desastre, 15)
fig, ax = plt.subplots(figsize=(8, 6))
words, counts_ = zip(*top_nd)
ax.barh(words[::-1], counts_[::-1], color='#4C72B0')
ax.set_title('Top 15 palabras mas frecuentes - No desastre')
ax.set_xlabel('Frecuencia')
plt.tight_layout()
plt.savefig(FIG_DIR + "06_top_palabras_no_desastre.png")
plt.close()

# ---------------------------------------------------------------
# 7. Top bigramas comparativo
# ---------------------------------------------------------------
top_bi_d = get_top_n(desastre, 10, (2, 2))
fig, ax = plt.subplots(figsize=(8, 6))
words, counts_ = zip(*top_bi_d)
ax.barh(words[::-1], counts_[::-1], color='#C44E52')
ax.set_title('Top 10 bigramas - Desastre real')
ax.set_xlabel('Frecuencia')
plt.tight_layout()
plt.savefig(FIG_DIR + "07_top_bigramas_desastre.png")
plt.close()

# ---------------------------------------------------------------
# 8. Longitud de tweets (palabras) por categoria
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
df[df['target'] == 1]['clean_word_count'].plot.hist(
    ax=ax, bins=20, alpha=0.6, color='#C44E52', label='Desastre real')
df[df['target'] == 0]['clean_word_count'].plot.hist(
    ax=ax, bins=20, alpha=0.6, color='#4C72B0', label='No desastre')
ax.set_title('Distribucion del numero de palabras por tweet (tras limpieza)')
ax.set_xlabel('Numero de palabras')
ax.set_ylabel('Frecuencia')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR + "08_longitud_tweets.png")
plt.close()

# ---------------------------------------------------------------
# 9. Top keywords por categoria (columna 'keyword' original)
# ---------------------------------------------------------------
kw = df.dropna(subset=['keyword'])
top_kw_d = kw[kw['target'] == 1]['keyword'].value_counts().head(10)
top_kw_nd = kw[kw['target'] == 0]['keyword'].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
top_kw_d.sort_values().plot.barh(ax=axes[0], color='#C44E52')
axes[0].set_title('Top 10 keywords - Desastre real')
top_kw_nd.sort_values().plot.barh(ax=axes[1], color='#4C72B0')
axes[1].set_title('Top 10 keywords - No desastre')
plt.tight_layout()
plt.savefig(FIG_DIR + "09_top_keywords.png")
plt.close()

print("Figuras generadas en", FIG_DIR)
import os
print(sorted(os.listdir(FIG_DIR)))
