from os import name

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

"""
NOTE: This is the non-example features.py for feature extraction purposes for the main model. For example features, see ex_features.py.

[ ] Word extraction
[ ] Spatial features

"""

nominal_features = [
    "house_type_name",
    "house_style",
    "municipality",
    "community",
]

numeric_features = [
    "square_feet",
    "bedrooms",
    "washrooms",
    "parking",
    "garage",
    "lot_front",
    "lot_depth",
    "lot_area",
]

# ---------------------------------------------------------------------------------------------------------------
# X feature extraction
# ---------------------------------------------------------------------------------------------------------------
def do_bag_of_words(df, column="remarks"):
    """Returns a DataFrame of binary word indicators."""
    vectorizer = CountVectorizer(
        binary=True,
        min_df=0.001,
        max_df=0.90,
        max_features=1000,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(df[column].fillna(""))

    return pd.DataFrame(
        matrix.toarray(),
        columns=["bow_" + w for w in vectorizer.get_feature_names_out()],
        index=df.index
    )

def tf_idf(df, column="remarks"):
    """Returns a DataFrame of TF-IDF features."""
    vectorizer = TfidfVectorizer(
        min_df=0.001,
        max_df=0.90,
        max_features=1000,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(df[column].fillna(""))

    return pd.DataFrame(
        matrix.toarray(),
        columns=["tfidf_" + w for w in vectorizer.get_feature_names_out()],
        index=df.index
    )

def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2

def sci_kit_learn_df(homes, state=None, use_remarks=True):
    """Features for homes with no list price."""

    fit = state is None

    if fit:
        state = {
            "vectorizer": TfidfVectorizer(
                min_df=0.002,
                max_df=0.90,
                max_features=1000,
                ngram_range=(1, 2),
                dtype=np.float32
            ),
            "scaler": StandardScaler()
        }

    X = pd.DataFrame(index=homes.index)

    # Numeric features
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]

    # Impute missing values using training statistics
    if fit:
        state["medians"] = X.median()

    X = X.fillna(state["medians"]).fillna(0)

    # TF-IDF features
    if use_remarks:
        remarks = homes["remarks"].fillna("").astype(str)

        if fit:
            text = state["vectorizer"].fit_transform(remarks)
        else:
            text = state["vectorizer"].transform(remarks)

        text = pd.DataFrame(
            text.toarray(),
            columns=[
                "tfidf_" + word
                for word in state["vectorizer"].get_feature_names_out()
            ],
            index=homes.index
        )

        X = pd.concat([X, text], axis=1)

    # Nominal features
    X = pd.concat([
        X,
        pd.get_dummies(
            homes[nominal_features].fillna("Unknown"),
            prefix=nominal_features,
            dtype=np.float32
        )
    ], axis=1)

    # Match training columns
    if fit:
        state["columns"] = X.columns.tolist()
    else:
        X = X.reindex(
            columns=state["columns"],
            fill_value=0
        )

    # Standardization
    if fit:
        values = state["scaler"].fit_transform(X)
    else:
        values = state["scaler"].transform(X)

    X = pd.DataFrame(
        values,
        columns=X.columns,
        index=X.index
    )

    p = homes[["longitude", "latitude"]].copy()
    t = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44

    return X, p, t, state


def list_aware_skl(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X, p, t, scaler = sci_kit_learn_df(homes)
    X["list_price"] = homes["list_price"]
    return X, p, t, scaler

def load_sample(df, n, random_state=None):
    """Load a dataset and return n random rows, optionally with a reproducible seed."""
    return df.sample(n=n, random_state=random_state)

if __name__ == "__main__":
    import data
    homes = data.load("listings")
    #print(homes.groupby("house_type_name")[["lot_front", "lot_depth"]].apply(lambda x: x.isna().sum()))


    X, scaler = sci_kit_learn_df(homes)
    print(X.columns.tolist())

    