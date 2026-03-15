"""
Text Preprocessing Pipeline for IMDB Sentiment Analysis
========================================================
Pipeline steps (applied in order):
    1. HTML tag removal
    2. Lowercasing
    3. Punctuation removal
    4. Stopword removal  (NLTK English stopwords)
    5. Lemmatization     (WordNetLemmatizer)
"""

# --- Importer ---
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Automatisk nedlasting av NLTK-data ---
# Kjøres én gang når fila importeres.
# Sjekker om stopwords, wordnet og omw-1.4 allerede er installert.
# Laster dem ned stille i bakgrunnen hvis de mangler.
for _pkg, _path in [("stopwords", "corpora/stopwords"),
                    ("wordnet",   "corpora/wordnet"),
                    ("omw-1.4",   "corpora/omw-1.4")]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# --- Globale variabler (lages én gang, gjenbrukes for alle 50 000 rader) ---
# Det er dyrt å opprette disse inne i en funksjon som kalles 50 000 ganger.
_STOP_WORDS  = set(stopwords.words("english"))       # ~180 engelske stoppord
_LEMMATIZER  = WordNetLemmatizer()                   # lemmatiseringsverktøy
_HTML_RE     = re.compile(r"<[^>]+>")                # regex som matcher HTML-tagger som <br />, <b>
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)  # tabell for å fjerne tegnsetting


def remove_html(text: str) -> str:
    # Erstatter alle HTML-tagger (<br />, <b>, <i> osv.) med et mellomrom.
    # " ".join(text.split()) fjerner doble mellomrom som kan oppstå.
    text = _HTML_RE.sub(" ", text)
    return " ".join(text.split())


def lowercase(text: str) -> str:
    # Gjør all tekst om til små bokstaver.
    return text.lower()


def remove_punctuation(text: str) -> str:
    # Fjerner alle tegnsetningstegn: . , ! ? " ' ( ) osv.
    # str.translate med _PUNCT_TABLE sletter tegn uten å erstatte dem.
    return text.translate(_PUNCT_TABLE)


def remove_stopwords(text: str) -> str:
    # Deler teksten inn i enkeltord (tokens).
    # Beholder bare ord som IKKE er i stoppordlisten.
    tokens = text.split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS)


def lemmatize(text: str) -> str:
    # Reduserer hvert ord til sin grunnform ved hjelp av en ordbok (WordNet).
    # Bedre enn stemming fordi resultatet alltid er et ekte ord.
    tokens = text.split()
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)


def preprocess(text: str) -> str:
    # Hovedfunksjonen — kjører alle fem steg
    # Stegene er avhengige av rekkefølgen:
    #   HTML må fjernes først (ellers kan < og > forstyrre resten)
    #   Lowercase før stopword-sjekk (stoppordlisten er lowercase)
    #   Punktum fjernes før lemmatisering (ellers feiltolkes "film." som eget token)
    text = remove_html(text)
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


if __name__ == "__main__":
    # Denne blokken kjøres KUN når fila startes direkte: python preprocessing.py
    # Når preprocessing.py importeres fra en annen fil, hoppes denne over.
    import os
    import pandas as pd
    from tqdm import tqdm

    tqdm.pandas()  # legger til progress_apply() på pandas Series

    base     = os.path.join(os.path.dirname(__file__), "..")
    in_path  = os.path.join(base, "data", "IMDB Dataset.csv")
    out_path = os.path.join(base, "data", "IMDB_preprocessed.csv")

    print("Loading data...")
    df = pd.read_csv(in_path)
    print(f"  {len(df)} reviews loaded")

    # Kaller preprocess() på hver rad i 'review'-kolonnen.
    # Resultatet lagres i en ny kolonne 'clean_review'.
    print("Preprocessing reviews (this takes a few minutes)...")
    df["clean_review"] = df["review"].progress_apply(preprocess)

    # Lagrer DataFrame med alle tre kolonner:
    #   review       — original tekst
    #   sentiment    — etikett (positive / negative)
    #   clean_review — ferdig prosessert tekst
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Columns: {df.columns.tolist()}")
