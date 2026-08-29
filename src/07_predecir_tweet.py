"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 7 (Ejercicio 7): Funcion de prediccion interactiva.
El usuario ingresa un tweet y el sistema lo clasifica como desastre real (1)
o no desastre (0).

Usa el mejor modelo y el vectorizador guardados por el paso 6
(../data/modelos/). Ejecutar desde la carpeta src/:

    python3 07_predecir_tweet.py

Para salir escriba 'q', 'salir' o 'exit'.
"""
import re
import numpy as np
import pandas as pd
import joblib
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Funcion de limpieza (identica a la del paso 2/6)
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
        text = text.replace('911', ' num911token ')
    text = PUNCT_RE.sub(' ', text)
    text = NUMBER_RE.sub(' ', text)
    if keep_911:
        text = text.replace('num911token', ' 911 ')
    text = MULTISPACE_RE.sub(' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Carga del modelo y el vectorizador guardados en el paso 6
# ---------------------------------------------------------------------------
MODELO = joblib.load("../data/modelos/mejor_modelo.joblib")
VECTORIZADOR = joblib.load("../data/modelos/vectorizer.joblib")
INFO = pd.read_csv("../data/modelos/info_modelo.csv").iloc[0]


def clasificar_tweet(texto: str) -> tuple:
    """Clasifica un tweet sin preprocesar.

    Devuelve (prediccion 0/1, confianza 0-1, texto limpio).
    """
    limpio = clean_text(texto)
    x = VECTORIZADOR.transform([limpio])
    pred = MODELO.predict(x)[0]
    conf = MODELO.predict_proba(x)[0][int(pred)]
    return int(pred), conf, limpio


def main():
    print("=" * 60)
    print("Clasificador de tweets - Desastre real (1) / No desastre (0)")
    print(f"Modelo: {INFO['mejor_modelo']} "
          f"(accuracy {INFO['accuracy_test']:.4f}, "
          f"F1 macro {INFO['f1_macro_test']:.4f})")
    print("Escriba un tweet y presione Enter. 'q' para salir.\n")
    print("=" * 60)
    while True:
        try:
            texto = input("tweet> ").strip()
        except EOFError:
            break
        if not texto:
            continue
        if texto.lower() in ("q", "salir", "exit", "quit"):
            print("Hasta luego.")
            break
        pred, conf, limpio = clasificar_tweet(texto)
        etiqueta = "DESASTRE" if pred == 1 else "no desastre"
        print(f"  limpio   : {limpio}")
        print(f"  resultado: {etiqueta} (confianza: {conf:.3f})")


if __name__ == "__main__":
    main()