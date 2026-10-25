"""Training-fitted text representations for controlled housing experiments.

The production 47-term TF-IDF path is intentionally unchanged. Experimental
variants wrap the non-text BoostingFeatures provider, keeping structured and
categorical preprocessing identical to the existing pipeline.

For full-vocabulary variants, terms are selected *without sold-price labels*
using document/term frequencies on the current training fold only. The same
selection rule and vocabulary order apply to TF-IDF, binary BoW, and count BoW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from features.gb_features import BoostingFeatures, tf_idf_features

TEXT_VARIANTS = (
    "no_text",
    "selected_tfidf",
    "selected_binary",
    "full_tfidf",
    "full_binary",
    "full_count",
)

SELECTED_TERMS = tuple(term.removeprefix("tfidf_") for term in tf_idf_features)
NGRAM_RANGE = (1, 2)


def training_vocabulary(remarks, *, max_features=1000, min_df=10):
    """Select a label-free unigram/bigram vocabulary on training remarks only."""
    if not isinstance(max_features, int) or max_features < 1:
        raise ValueError("max_features must be a positive integer")
    if not isinstance(min_df, int) or min_df < 1:
        raise ValueError("min_df must be a positive integer")
    documents = pd.Series(remarks).fillna("").astype(str)
    selector = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=NGRAM_RANGE,
        dtype=np.float32,
    )
    return tuple(selector.fit(documents).get_feature_names_out())


class TextAugmentedFeatures:
    """Append a count, binary-presence, or TF-IDF representation to base features.

    \x60terms=None\x60 with vocabulary_mode='full' means build a vocabulary using
    training remarks only. Passing an explicit training-fitted \x60terms\x60 ensures
    an identical vocabulary across different representations.
    """

    def __init__(
        self,
        base_provider,
        *,
        representation,
        vocabulary_mode="full",
        terms=None,
        max_features=1000,
        min_df=10,
    ):
        if representation not in ("tfidf", "binary", "count"):
            raise ValueError("representation must be tfidf, binary, or count")
        if vocabulary_mode not in ("selected", "full"):
            raise ValueError("vocabulary_mode must be selected or full")
        self.base_provider = base_provider
        self.representation = representation
        self.vocabulary_mode = vocabulary_mode
        self.terms = tuple(terms) if terms is not None else None
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer_ = None
        self.text_columns_ = None
        self.columns_ = None

    @staticmethod
    def _remarks(homes):
        if "remarks" not in homes.columns:
            return pd.Series("", index=homes.index, dtype=str)
        return homes["remarks"].fillna("").astype(str)

    def _fit_text(self, homes):
        documents = self._remarks(homes)
        if self.terms is not None:
            vocabulary = self.terms
        elif self.vocabulary_mode == "selected":
            vocabulary = SELECTED_TERMS
        else:
            vocabulary = training_vocabulary(
                documents,
                max_features=self.max_features,
                min_df=self.min_df,
            )
        if not vocabulary:
            raise ValueError("No text terms survived vocabulary selection")
        if self.representation == "tfidf":
            vectorizer = TfidfVectorizer(
                vocabulary=list(vocabulary),
                ngram_range=NGRAM_RANGE,
                dtype=np.float32,
            )
        else:
            vectorizer = CountVectorizer(
                vocabulary=list(vocabulary),
                ngram_range=NGRAM_RANGE,
                binary=self.representation == "binary",
                dtype=np.float32,
            )
        self.vectorizer_ = vectorizer.fit(documents)
        self.text_columns_ = [
            f"text_{i}" for i in range(len(self.vectorizer_.get_feature_names_out()))
        ]

    def _text_frame(self, homes):
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit_transform before transform")
        matrix = self.vectorizer_.transform(self._remarks(homes))
        return pd.DataFrame(
            matrix.toarray(), index=homes.index, columns=self.text_columns_
        )

    def fit_transform(self, homes, model="catboost", track="off_market"):
        self._fit_text(homes)
        base = self.base_provider.fit_transform(homes, model=model, track=track)
        self.columns_ = list(base.columns) + self.text_columns_
        return pd.concat([base, self._text_frame(homes)], axis=1)

    def transform(self, homes, model="catboost"):
        if self.columns_ is None:
            raise RuntimeError("Call fit_transform before transform")
        base = self.base_provider.transform(homes, model=model)
        return pd.concat([base, self._text_frame(homes)], axis=1)[self.columns_]

    @property
    def categorical_features(self):
        return self.base_provider.categorical_features

    @property
    def text_feature_names(self):
        if self.vectorizer_ is None:
            return {}
        return dict(zip(self.text_columns_, self.vectorizer_.get_feature_names_out()))


def make_text_base(
    variant, *, include_time=True, full_vocabulary=None,
    max_features=1000, min_df=10
):
    """Return base features for a variant, with unchanged non-text columns."""
    if variant not in TEXT_VARIANTS:
        raise ValueError(f"Unknown text variant: {variant}")
    if variant == "selected_tfidf":
        # Exact original pathway; preserves legacy TF-IDF, column names and IDF.
        return BoostingFeatures(use_remarks=True, include_time=include_time)
    base = BoostingFeatures(use_remarks=False, include_time=include_time)
    if variant == "no_text":
        return base
    mode, representation = variant.split("_", 1)
    return TextAugmentedFeatures(
        base,
        representation=representation,
        vocabulary_mode="selected" if mode == "selected" else "full",
        terms=full_vocabulary if mode == "full" else None,
        max_features=max_features,
        min_df=min_df,
    )
