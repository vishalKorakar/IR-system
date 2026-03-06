import nltk
nltk.download('punkt')
nltk.download('stopwords')

import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


class Preprocessor:
    def __init__(self, use_stemming=False):
        self.use_stemming = use_stemming

    # ------------------------------------------------------------
    # Boilerplate Removal
    # ------------------------------------------------------------
    def remove_gutenberg_boilerplate(self, text):
        start_pattern = r"\*\*\* START OF.*?\*\*\*"
        end_pattern = r"\*\*\* END OF.*?\*\*\*"

        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        end_match = re.search(end_pattern, text, re.IGNORECASE | re.DOTALL)

        if start_match and end_match:
            return text[start_match.end():end_match.start()].strip()

        lines = text.split("\n")
        return "\n".join(lines[100:]).strip()

    # ------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------
    def normalize_unicode(self, text):
        return unicodedata.normalize("NFC", text)

    def lowercase(self, text):
        return text.lower()

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_punctuation(self, tokens):
        return [t for t in tokens if re.search(r'\w', t)]

    def apply_stemming(self, tokens):
        return [STEMMER.stem(t) for t in tokens]

    # ------------------------------------------------------------
    # BODY preprocessing (keep stopwords for phrase correctness)
    # ------------------------------------------------------------
    def preprocess_body(self, raw_text):
        text = self.remove_gutenberg_boilerplate(raw_text)
        text = self.normalize_unicode(text)
        text = self.lowercase(text)

        tokens = self.tokenize(text)
        tokens = self.remove_punctuation(tokens)

        if self.use_stemming:
            tokens = self.apply_stemming(tokens)

        return tokens

    # ------------------------------------------------------------
    # FIELD preprocessing (title/author)
    # Keep stopwords OFF by default for clean structured search
    # ------------------------------------------------------------
    def preprocess_field(self, value):
        if not value:
            return []

        text = self.normalize_unicode(value)
        text = self.lowercase(text)

        tokens = self.tokenize(text)
        tokens = self.remove_punctuation(tokens)

        # For structured fields, remove stopwords
        tokens = [t for t in tokens if t not in STOPWORDS]

        if self.use_stemming:
            tokens = self.apply_stemming(tokens)

        return tokens