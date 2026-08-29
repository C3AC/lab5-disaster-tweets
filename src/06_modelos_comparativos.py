"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 6 (continuacion): Comparacion de multiples modelos de clasificacion
para determinar si un tweet se refiere a un desastre real o no.

Enfoque para abordar el contexto:
- Representacion TF-IDF de unigramas + bigramas (ngram_range=(1,2)),
  5000 caracteristicas (igual que el baseline), para capturar contexto local.
- Validacion cruzada estratificada (5 folds) para comparar modelos de forma
  robusta sin depender de una sola particion.
- Modelos probados: Regresion Logistica, Naive Bayes Multinomial, SVM lineal
  (LinearSVC), Random Forest y KNN (baseline de distancia).
- Ajuste de hiperparametros (GridSearchCV) para los mejores candidatos.
- Seleccion del mejor modelo por F1 macro en el conjunto de prueba.
- Curva ROC, guardado del modelo y funcion de prediccion para tweets nuevos.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Funcion de limpieza (identica a la del paso 2), necesaria para predecir
# tweets nuevos que no estan preprocesados.
# ---------------------------------------------------------------------------
STOPWORDS = set(stopwords.words('english'))
NEGATIONS = {"no", "not", "nor"}
STOPWORDS = STOPWORDS - NEGATIONS

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_SYMBOL_RE = re.compile(r'#')
HTML_ENTITY_RE = re.compile(r'&\w+;')
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+", flags=re.UNICODE
)
PUNCT_RE = re.compile(r"[^\w\s]")
NUMBER_RE = re.compile(r'\b\d+\b')
MULTISPACE_RE = re.compile(r'\s+')


def clean_text(text: str, keep_911: bool = True) -> str:
    text = str(text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = HASHTAG_SYMBOL_RE.sub('', text)
    text = EMOJI_RE.sub(' ', text)
    text = HTML_ENTITY_RE.sub(' ', text)
    text = text.replace('_', ' ')
    if keep_911:
        text = text.replace('911', ' NUM911TOKEN ')
    text = PUNCT_RE.sub(' ', text)
    text = NUMBER_RE.sub(' ', text)
    if keep_911:
        text = text.replace('num911token', ' 911 ')
    text = MULTISPACE_RE.sub(' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Carga y particion (misma configuracion que el baseline para ser comparable)
# ---------------------------------------------------------------------------
df = pd.read_csv("../data/train_clean.csv")
df = df.dropna(subset=['text_clean'])
df = df[df['text_clean'].str.len() > 0]

X = df['text_clean']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Entrenamiento: {len(X_train)} tweets | Prueba: {len(X_test)} tweets",
      flush=True)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# KNN es muy costoso en validacion cruzada sobre TF-IDF de alta dimension,
# por lo que se excluye del CV y solo se evalua sobre el conjunto de prueba.
MODELOS_CV = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes Multinomial": MultinomialNB(),
    "SVM lineal (LinearSVC)": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                            n_jobs=-1),
}
MODELOS_TOTAL = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes Multinomial": MultinomialNB(),
    "SVM lineal (LinearSVC)": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                            n_jobs=-1),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, algorithm='brute',
                                      n_jobs=-1),
}

SCORING = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# 1. Comparacion con validacion cruzada (5 folds)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDACION CRUZADA (5 folds, datos de entrenamiento)")
print("=" * 70, flush=True)
filas_cv = []
for nombre, modelo in MODELOS_CV.items():
    cv_res = cross_validate(modelo, X_train_tfidf, y_train, cv=cv,
                            scoring=SCORING, n_jobs=-1)
    fila = {"modelo": nombre}
    for m in SCORING:
        fila[m] = cv_res[f"test_{m}"].mean().round(4)
        fila[m + "_std"] = cv_res[f"test_{m}"].std().round(4)
    filas_cv.append(fila)
    print(f"\n{nombre}", flush=True)
    for m in SCORING:
        print(f"  {m:18s}: {fila[m]:.4f} +/- {fila[m+'_std']:.4f}")

cv_df = pd.DataFrame(filas_cv)
cv_df.to_csv("../data/resultados_cv_modelos.csv", index=False)
print("\nCV guardado en ../data/resultados_cv_modelos.csv", flush=True)

# Grafico de validacion cruzada
fig, ax = plt.subplots(figsize=(10, 6))
cv_df.set_index('modelo')[['accuracy', 'precision_macro', 'recall_macro',
                           'f1_macro']].plot.bar(ax=ax, width=0.8)
ax.set_title('Comparacion de modelos con validacion cruzada (5 folds)')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("../figures/12_cv_comparacion_modelos.png")
plt.show()

# ---------------------------------------------------------------------------
# 2. Ajuste de hiperparametros (GridSearchCV, 3 folds)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("AJUSTE DE HIPERPARAMETROS (GridSearchCV, 3 folds)")
print("=" * 70, flush=True)
gcv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
base_estimadores = {
    "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "SVM lineal (LinearSVC)": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
}
param_grids = {
    "Regresion Logistica": {"C": [0.1, 1, 10]},
    "SVM lineal (LinearSVC)": {"C": [0.1, 1, 10]},
    "Random Forest": {"n_estimators": [200, 300]},
}

mejores_modelos = {}
for nombre, base in base_estimadores.items():
    gs = GridSearchCV(base, param_grids[nombre], scoring='f1_macro', cv=gcv,
                      n_jobs=-1)
    gs.fit(X_train_tfidf, y_train)
    mejores_modelos[nombre] = gs.best_estimator_
    print(f"{nombre} -> mejores hiperparametros: {gs.best_params_} "
          f"| F1 macro (CV): {gs.best_score_:.4f}", flush=True)

# Modelos que no se ajustan: se usan tal cual
for nombre in MODELOS_CV:
    if nombre not in mejores_modelos:
        mejores_modelos[nombre] = MODELOS_CV[nombre]
mejores_modelos["KNN (k=5)"] = MODELOS_TOTAL["KNN (k=5)"]

# ---------------------------------------------------------------------------
# 3. Evaluacion final sobre el conjunto de prueba
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EVALUACION FINAL SOBRE EL CONJUNTO DE PRUEBA (20%)")
print("=" * 70, flush=True)

def evaluar(modelo, X_te, y_te):
    y_pred = modelo.predict(X_te)
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average='macro'),
        "recall_macro": recall_score(y_te, y_pred, average='macro'),
        "f1_macro": f1_score(y_te, y_pred, average='macro'),
        "f1_weighted": f1_score(y_te, y_pred, average='weighted'),
    }

filas_test = []
for nombre, modelo in mejores_modelos.items():
    modelo.fit(X_train_tfidf, y_train)
    y_pred = modelo.predict(X_test_tfidf)
    m = evaluar(modelo, X_test_tfidf, y_test)
    m["modelo"] = nombre
    filas_test.append(m)
    acc = m["accuracy"]
    print(f"\n{nombre} | acc={acc:.4f} | f1_macro={m['f1_macro']:.4f}",
          flush=True)
    print(classification_report(y_test, y_pred,
                                target_names=['No desastre', 'Desastre real']))

test_df = pd.DataFrame(filas_test).set_index('modelo')
test_df = test_df[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                   'f1_weighted']]
print("\nResumen comparativo en prueba:")
print(test_df.round(4))
test_df.round(4).to_csv("../data/resultados_modelos_finales.csv")

# ---------------------------------------------------------------------------
# 4. Seleccion del mejor modelo por F1 macro en prueba
# ---------------------------------------------------------------------------
mejor_nombre = test_df['f1_macro'].idxmax()
mejor_modelo = mejores_modelos[mejor_nombre]
print("\n" + "=" * 70)
print(f"MEJOR MODELO: {mejor_nombre}")
print(f"F1 macro en prueba: {test_df.loc[mejor_nombre, 'f1_macro']:.4f}")
print(f"Accuracy en prueba: {test_df.loc[mejor_nombre, 'accuracy']:.4f}")
print("=" * 70, flush=True)

y_pred_mejor = mejor_modelo.predict(X_test_tfidf)

# Matriz de confusion del mejor modelo
cm = confusion_matrix(y_test, y_pred_mejor)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No desastre', 'Desastre'],
            yticklabels=['No desastre', 'Desastre'])
ax.set_title(f'Matriz de confusion - {mejor_nombre}')
ax.set_ylabel('Real')
ax.set_xlabel('Predicho')
plt.tight_layout()
plt.savefig("../figures/13_matriz_confusion_mejor.png")
plt.show()

# Curva ROC del mejor modelo
if hasattr(mejor_modelo, "predict_proba"):
    proba = mejor_modelo.predict_proba(X_test_tfidf)[:, 1]
else:
    proba = mejor_modelo.decision_function(X_test_tfidf)
roc_auc = roc_auc_score(y_test, proba)
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_title(f'Curva ROC - {mejor_nombre}')
ax.set_xlabel('Tasa de falsos positivos')
ax.set_ylabel('Tasa de verdaderos positivos')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("../figures/14_roc_mejor_modelo.png")
plt.show()

# ---------------------------------------------------------------------------
# 5. Guardado del modelo, vectorizador y metadatos
# ---------------------------------------------------------------------------
os.makedirs("../data/modelos", exist_ok=True)
joblib.dump(mejor_modelo, "../data/modelos/mejor_modelo.joblib")
joblib.dump(vectorizer, "../data/modelos/vectorizer.joblib")
info = pd.DataFrame([{
    "mejor_modelo": mejor_nombre,
    "f1_macro_test": test_df.loc[mejor_nombre, 'f1_macro'],
    "accuracy_test": test_df.loc[mejor_nombre, 'accuracy'],
    "roc_auc_test": roc_auc,
    "max_features": 5000,
    "ngram_range": "(1, 2)",
}])
info.to_csv("../data/modelos/info_modelo.csv", index=False)
print("\nModelo guardado en ../data/modelos/", flush=True)

# ---------------------------------------------------------------------------
# 6. Funcion de prediccion para tweets nuevos
# ---------------------------------------------------------------------------
def predecir_tweet(texto: str) -> tuple:
    """Recibe un tweet sin preprocesar y devuelve la clasificacion."""
    limpio = clean_text(texto)
    x = vectorizer.transform([limpio])
    pred = mejor_modelo.predict(x)[0]
    if hasattr(mejor_modelo, "predict_proba"):
        confianza = mejor_modelo.predict_proba(x)[0][1]
    else:
        confianza = float(np.clip(mejor_modelo.decision_function(x)[0], 0, 1))
    return int(pred), confianza, limpio


print("\n" + "=" * 70)
print("EJEMPLOS DE PREDICCION CON TWEETS NUEVOS")
print("=" * 70, flush=True)
ejemplos = [
    "Forest fire near La Ronge Sask. Canada",
    "I love this new phone, it's fire!",
    "BREAKING: 7.1 magnitude earthquake hits Mexico City",
    "That concert was the bomb, best night ever",
]
for t in ejemplos:
    pred, conf, limpio = predecir_tweet(t)
    etiqueta = "DESASTRE" if pred == 1 else "no desastre"
    print(f"\nTweet: {t}")
    print(f"Limpio: {limpio}")
    print(f"Clasificacion: {etiqueta} (confianza: {conf:.3f})")