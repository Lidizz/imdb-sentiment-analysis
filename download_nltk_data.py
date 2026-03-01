"""
Run this script once after setting up the virtual environment.
It downloads the NLTK data packages required by the preprocessing pipeline.

Usage:
    python download_nltk_data.py
"""
import nltk

packages = [
    ('stopwords', 'corpora/stopwords'),
    ('wordnet',   'corpora/wordnet'),
    ('omw-1.4',   'corpora/omw-1.4'),
]

print("Downloading NLTK data...\n")
for name, path in packages:
    try:
        nltk.data.find(path)
        print(f"  [already installed] {name}")
    except LookupError:
        print(f"  [downloading]       {name}")
        nltk.download(name, quiet=True)

print("\nDone. NLTK data is ready.")
