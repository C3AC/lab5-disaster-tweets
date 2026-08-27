"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 5 (avance): Modelo preliminar de clasificacion

Enfoque para abordar el contexto:
- Se representa cada tweet limpio con TF-IDF de unigramas y bigramas
  (ngram_range=(1,2)) para capturar algo de contexto local (ej. "suicide
  bomber" es mas informativo que "suicide" y "bomber" por separado).
- Se limita el vocabulario a las 5000 caracteristicas mas relevantes
  (max_features) para reducir dimensionalidad y ruido.
- Como modelo preliminar/baseline se usa Regresion Logistica y Naive Bayes
  Multinomial, dos algoritmos simples, rapidos y ampliamente usados como
  punto de partida en clasificacion de texto. En la entrega final se
  compararan con mas algoritmos (SVM, Random Forest, etc.) y se seleccionara
  el mejor segun metricas.
- Se separa 80% entrenamiento / 20% prueba, de forma estratificada para
  mantener la proporcion de clases.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/train_clean.csv")
df = df.dropna(subset=['text_clean'])
df = df[df['text_clean'].str.len() > 0]

X = df['text_clean']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Entrenamiento: {len(X_train)} tweets | Prueba: {len(X_test)} tweets")

# TF-IDF con unigramas + bigramas para capturar contexto
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

modelos = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes Multinomial": MultinomialNB(),
}

resultados = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train_tfidf, y_train)
    y_pred = modelo.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}\n{nombre}\n{'='*50}")
    print(classification_report(y_test, y_pred,
                                 target_names=['No desastre', 'Desastre real']))

    resultados.append({
        "modelo": nombre, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1
    })

    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No desastre', 'Desastre'],
                yticklabels=['No desastre', 'Desastre'])
    ax.set_title(f'Matriz de confusion - {nombre}')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')
    plt.tight_layout()
    safe_name = nombre.lower().replace(' ', '_')
    plt.savefig(f"../figures/10_confusion_{safe_name}.png")
    plt.close()

res_df = pd.DataFrame(resultados)
print("\nResumen comparativo de modelos preliminares:")
print(res_df.round(4))
res_df.to_csv("../data/resultados_modelo_preliminar.csv", index=False)

# Grafico comparativo de metricas
fig, ax = plt.subplots(figsize=(8, 5))
res_df.set_index('modelo')[['accuracy', 'precision', 'recall', 'f1']].plot.bar(ax=ax)
ax.set_title('Comparacion de modelos preliminares')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("../figures/11_comparacion_modelos.png")
plt.close()

print("\nModelos y resultados guardados.")
