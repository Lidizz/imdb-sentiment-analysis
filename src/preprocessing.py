"""Text preprocessing pipeline for IMDB sentiment analysis."""
import re
from functools import lru_cache
from typing import Callable, Iterable, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag, pos_tag_sents
from nltk.stem import WordNetLemmatizer

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

_HTML_TAG_RE = re.compile(r'<.*?>')
_SPECIAL_CHAR_RE = re.compile(r'[^a-zA-Z\s]')

_NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'nobody',
    'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'
}

_STOP_WORDS = set(stopwords.words('english')) - _NEGATION_WORDS  # preserve negation; "not good" != "good"
_LEMMATIZER = WordNetLemmatizer()

def _to_wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet POS tags"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    return _HTML_TAG_RE.sub('', text)

def remove_special_characters(text: str) -> str:
    """Keep only letters and spaces. Remove punctuation, numbers, etc."""
    return _SPECIAL_CHAR_RE.sub('', text)

def to_lowercase(text: str) -> str:
    """Convert to lowercase for consistent vocabulary"""
    return text.lower()

def remove_stopwords(text: str) -> str:
    """Remove common English words that carry no sentiment signal"""
    return ' '.join([w for w in text.split() if w not in _STOP_WORDS])


@lru_cache(maxsize=200_000)
def _lemmatize_word(word: str, wn_pos: str) -> str:
    """Cache per-token lemmatization to speed repeated words"""
    return _LEMMATIZER.lemmatize(word, wn_pos)

def lemmatize_text(text: str) -> str:
    """Reduce words to base form using POS-aware lemmatization"""
    tokens = text.split()
    if not tokens:
        return text

    tagged_words = pos_tag(tokens)
    return ' '.join(
        [_lemmatize_word(word, _to_wordnet_pos(pos)) for word, pos in tagged_words]
    )

def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline:

    1. HTML removal       - IMDB-specific noise (signal vs noise)
    2. Lowercase          - 'Good' and 'good' are the same word
    3. Special chars      - punctuation doesn't help bag-of-words models
    4. Stopword removal   - 'the', 'is', 'at' carry no sentiment
    5. Lemmatization      - reduces vocabulary size, groups related words
    """
    text = remove_html_tags(text)
    text = to_lowercase(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_texts_batch(
    texts: Iterable[str],
    batch_size: int = 1000,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[str]:
    """Preprocess using POS tagging in batches to speed up large dataset processing"""
    text_list = [str(text) for text in texts]
    if not text_list:
        return []

    cleaned = []
    for text in text_list:
        text = remove_html_tags(text)
        text = to_lowercase(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        cleaned.append(text)

    outputs: List[str] = []
    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start:start + batch_size]
        token_sents = [text.split() for text in batch]
        tagged_sents = pos_tag_sents(token_sents)

        for tagged_words in tagged_sents:
            if not tagged_words:
                outputs.append('')
                if progress_callback is not None:
                    progress_callback(1)
                continue
            lemma_tokens = [
                _lemmatize_word(word, _to_wordnet_pos(pos))
                for word, pos in tagged_words
            ]
            outputs.append(' '.join(lemma_tokens))
            if progress_callback is not None:
                progress_callback(1)

    return outputs