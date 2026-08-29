"""
Laboratorio 5 - Mineria de Textos y Analisis de Sentimiento
CC3084 - Data Science - UVG

Paso 2: Limpieza y preprocesamiento de texto

Tareas realizadas (documentadas para el informe):
1. Conversion a minusculas
2. Eliminacion de URLs (http, https, www)
3. Eliminacion de menciones (@usuario) y hashtags -> se conserva la palabra del
   hashtag pero se quita el simbolo "#", porque suele llevar informacion util
   (ej. "#earthquake" -> "earthquake")
4. Eliminacion de emoticones/emojis
5. Eliminacion de codigos HTML (&amp;, etc.)
6. Eliminacion de signos de puntuacion
7. Manejo de numeros: se eliminan los numeros en general, PERO se conserva
   "911" como token especial porque es semanticamente relevante para
   identificar tweets de desastres reales (llamadas de emergencia)
8. Eliminacion de stopwords (articulos, preposiciones, conjunciones) en ingles,
   ya que el dataset esta en ingles
9. Eliminacion de espacios extra
10. Tokenizacion
11. Correccion de "mojibake": el dataset original tiene texto mal codificado
    (secuencias como "\x89Û_", "\x89ÛÒ") producto de caracteres especiales
    (comillas curvas, elipsis) leidos con una codificacion incorrecta. Se
    eliminan estos caracteres no-ASCII ya que no aportan informacion util y
    contaminaban las frecuencias de palabras (aparecia "u_" como top word).
"""
import re
import pandas as pd
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
# Conservamos negaciones relevantes para analisis de sentimiento posterior
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
    """Limpia un tweet siguiendo los pasos documentados arriba."""
    text = str(text)

    # 0. corregir mojibake / caracteres no-ASCII (comillas curvas, elipsis mal
    #    codificadas que aparecen como \x89Û_, \x89ÛÒ, etc. en este dataset)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 1. minusculas
    text = text.lower()

    # 2. URLs
    text = URL_RE.sub(' ', text)

    # 3. menciones
    text = MENTION_RE.sub(' ', text)

    # 3b. simbolo de hashtag (se conserva la palabra)
    text = HASHTAG_SYMBOL_RE.sub('', text)

    # 4. emoticones/emojis
    text = EMOJI_RE.sub(' ', text)

    # 5. entidades HTML
    text = HTML_ENTITY_RE.sub(' ', text)

    # 5b. guiones bajos residuales de mojibake truncado (ej. "rea_" viene de
    #     "read\x89\xdb_" al quitar los bytes no-ASCII) -> se tratan como
    #     puntuacion, no como parte de la palabra
    text = text.replace('_', ' ')

    # 6. proteger "911" antes de quitar numeros/puntuacion
    if keep_911:
        text = text.replace('911', ' num911token ')

    # 7. apostrofes y puntuacion
    text = PUNCT_RE.sub(' ', text)

    # 8. numeros restantes
    text = NUMBER_RE.sub(' ', text)

    if keep_911:
        text = text.replace('num911token', ' 911 ')

    # 9. espacios extra
    text = MULTISPACE_RE.sub(' ', text).strip()

    # 10. stopwords
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]

    return ' '.join(tokens)


if __name__ == "__main__":
    df = pd.read_csv("../data/train_explored.csv")

    # Eliminar duplicados exactos de texto (mismo texto, mismo target)
    before = len(df)
    df = df.drop_duplicates(subset=['text', 'target']).reset_index(drop=True)
    print(f"Filas eliminadas por duplicado exacto (texto+target): {before - len(df)}")

    df['text_clean'] = df['text'].apply(clean_text)
    df['clean_word_count'] = df['text_clean'].str.split().str.len()

    print("\nEjemplos antes/despues de limpieza:")
    for i in range(5):
        print(f"\nOriginal: {df['text'].iloc[i]}")
        print(f"Limpio  : {df['text_clean'].iloc[i]}")

    print("\nFilas que quedaron con texto vacio tras limpieza:",
          (df['text_clean'].str.len() == 0).sum())

    df.to_csv("../data/train_clean.csv", index=False)
    print("\nGuardado: ../data/train_clean.csv")
