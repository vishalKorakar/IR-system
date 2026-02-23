import nltk
nltk.download('punkt')
nltk.download('stopwords')

import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize once
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


class Preprocessor:
    def __init__(self, use_stemming=False):
        """
        use_stemming: Boolean flag to enable/disable stemming.
        """
        self.use_stemming = use_stemming
        self.stemmer = nltk.stem.PorterStemmer() if use_stemming else None


    def remove_gutenberg_boilerplate(self, text):
        """
        Extract content between START/END markers if present.
        Falls back to raw text if markers aren't found.
        """
        start_pattern = r"\*\*\* START OF.*?\*\*\*" # .* means - Delete all the text after START OF 
        end_pattern = r"\*\*\* END OF.*?\*\*\*"

        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        end_match = re.search(end_pattern, text, re.IGNORECASE | re.DOTALL)

        if start_match and end_match:
            return text[start_match.end():end_match.start()].strip()

        # Fallback: remove first ~500 lines heuristically
        lines = text.split("\n")
        return "\n".join(lines[100:]).strip()


    def normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode text (preserves accented characters).
        """
        return unicodedata.normalize("NFC", text)

    def lowercase(self, text: str) -> str:
        """
        Case normalization.
        """
        return text.lower()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes text using NLTK's Punkt tokenizer.
        """
        return word_tokenize(text)

    def remove_punctuation(self, tokens: list[str]) -> list[str]:
        """
        Removes tokens that are purely punctuation.
        """
        return [t for t in tokens if re.search(r'\w', t)]

    # def remove_stopwords(self, tokens: list[str]) -> list[str]:
    #     """
    #     Removes common English stopwords.
    #     """
    #     return [t for t in tokens if t not in STOPWORDS]

    def apply_stemming(self, tokens: list[str]) -> list[str]:
        """
        Applies Porter stemming.
        """
        return [STEMMER.stem(t) for t in tokens]

    def preprocess(self, raw_text: str) -> list[str]:
        """
        Full preprocessing pipeline.
        """
        text = self.remove_gutenberg_boilerplate(raw_text)
        text = self.normalize_unicode(text)
        text = self.lowercase(text)

        tokens = self.tokenize(text)
        tokens = self.remove_punctuation(tokens)
        # tokens = self.remove_stopwords(tokens) Not done for keeping stopwords for term search

        if self.use_stemming:
            tokens = self.apply_stemming(tokens)

        return tokens

    # def preprocess_field(self, value: str) -> List[str]:
    #         """
    #         Preprocess small fields like title/author:
    #         - normalize unicode
    #         - tokenize
    #         - remove punctuation
    #         - (optionally) stopwords (kept ON for simplicity; can be OFF if desired)
    #         """
    #         if not value:
    #             return []
    #         value = self.normalize_unicode(value)
    #         tokens = self.tokenize(value)
    #         tokens = self.remove_punctuation(tokens)
    #         tokens = self.remove_stopwords(tokens)  # You may disable for author names if you prefer
    #         if self.use_stemming:
    #             tokens = self.apply_stemming(tokens)
    #         return tokens