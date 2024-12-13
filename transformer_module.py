# Data manipulation and utilities
import os
import io
import logging
import re
import string
import emoji
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter

# Machine learning and model utilities
from sklearn.base import BaseEstimator, TransformerMixin
import joblib  # To load the .pkl model file




# NLP libraries
from underthesea import word_tokenize, text_normalize


class VietnameseTextNormalize(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [text_normalize(s) for s in x]
    
    def fit(self, x, y=None):
        return self

class VietnameseWordTokenizer(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [' '.join(word_tokenize(s)) for s in x]
    
    def fit(self, x, y=None):
        return self
    
    
# Reuse the custom transformers (same as before)
class RemoveConsecutiveSpaces(BaseEstimator, TransformerMixin):
    def remove_consecutive_spaces(self, s):
        return ' '.join(s.split())

    def transform(self, x):
        return [self.remove_consecutive_spaces(s) for s in x]

    def fit(self, x, y=None):
        return self


class RemovePunct(BaseEstimator, TransformerMixin):
    non_special_chars = re.compile('[^A-Za-z0-9 ]+')

    def remove_punct(self, s):
        return re.sub(self.non_special_chars, '', s)

    def transform(self, x):
        return [self.remove_punct(s) for s in x]

    def fit(self, x, y=None):
        return self


class Lowercase(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [s.lower() for s in x]

    def fit(self, x, y=None):# Data manipulation and utilities
        return self





class NumWordsCharsFeature(BaseEstimator, TransformerMixin):
    def count_char(self, s):
        return len(s)

    def count_word(self, s):
        return len(s.split())

    def transform(self, x):
        count_chars = sp.csr_matrix([self.count_char(s) for s in x], dtype=np.float64).transpose()
        count_words = sp.csr_matrix([self.count_word(s) for s in x], dtype=np.float64).transpose()
        return sp.hstack([count_chars, count_words])

    def fit(self, x, y=None):
        return self


class ExclamationMarkFeature(BaseEstimator, TransformerMixin):
    def count_exclamation(self, s):
        count = s.count('!') + s.count('?')
        return count / (1 + len(s.split()))

    def transform(self, x):
        counts = [self.count_exclamation(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class RemoveEmoji(BaseEstimator, TransformerMixin):
    def remove_emoji(self, s):
        return ''.join(c for c in s if c not in emoji.EMOJI_DATA)

    def transform(self, x):
        return [self.remove_emoji(s) for s in x]

    def fit(self, x, y=None):
        return self



class NumCapitalLettersFeature(BaseEstimator, TransformerMixin):
    def count_upper(self, s):
        n_uppers = sum(1 for c in s if c.isupper())
        return n_uppers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumLowercaseLettersFeature(BaseEstimator, TransformerMixin):
    def count_lower(self, s):
        n_lowers = sum(1 for c in s if c.islower())
        return n_lowers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_lower(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumPunctsFeature(BaseEstimator, TransformerMixin):
    def count(self, l1, l2):
        return sum([1 for x in l1 if x in l2])

    def count_punct(self, s):
        n_puncts = self.count(s, set(string.punctuation))
        return n_puncts / (1 + len(s))

    def transform(self, x):
        counts = [self.count_punct(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self






