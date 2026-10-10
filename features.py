import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
NOTE: This is the non-example features.py for feature extraction purposes for the main model. For example features, see ex_features.py.

[ ] Word extraction
[ ] Spatial features

"""

nominal_features = ['house_type_name', 'house_style', 'municipality', 'community']


def do_bag_of_words(df, column="remarks"):
    """A simple bag-of-words representation of the remarks text.
    Returns a table with one row per home and one column per word.
    Each cell is 1 if the word appears in that home's remarks, else 0.
    """
    # Split each remark into words, lowercase them, and remove punctuation.
    remarks = df[column]
    words = remarks.str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.split()
    # Get the set of all words that appear anywhere.
    vocab = sorted(set(word for row in words for word in row))
    # Replaces remarks with columns for each word, with 1 if the word appears in that home's remarks, else 0.
    bag_of_words = pd.DataFrame(0, index=df.index, columns=vocab)
    # Fill in 1s where the word appears in that home's remarks.
    for i, row in words.items():
        bag_of_words.loc[i, row] = 1
    return bag_of_words

def tf_idf(df, column="remarks"):
    """
    A simple TF-IDF representation of the remarks text.
    Returns a table with one row per home and one column per word.
    Each cell is the TF-IDF score for that word in that home's remarks.
    TF-IDF is a measure of how important a word is to a document in a collection
    """
    bow = do_bag_of_words(df, column)
    # Term frequency (TF): number of times each word appears in each home's remarks.
    tf = bow.div(bow.sum(axis=1), axis=0)
    # Inverse document frequency (IDF): log of the total number of homes divided by the number of homes that contain each word.
    idf = np.log(len(df) / (bow > 0).sum(axis=0))
    # TF-IDF: multiply TF by IDF for each word.
    tf_idf = tf.mul(idf, axis=1)
    return tf_idf

def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def spatial_features(df, target_col, lat_col="latitude", lon_col="longitude"):
    """Creates spatial interaction features based on latitude and longitude."""
    return df[[lat_col, lon_col]].copy()


def off_market_features(homes):
    """Features for homes with no list price.
    NOTE: Does not include most variable
    """
    X = pd.DataFrame(index=homes.index)

    """
    Numeric features
    """

    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]


    #X = X.concat([X, do_bag_of_words(homes, column="remarks")], axis=1)
    X = X.concat([X, tf_idf(homes, column="remarks")], axis=1)

    
    # One 0/1 column per nominal feature value
    for feature in nominal_features:
        for kind in homes[feature].unique():
            X[f"{feature}_{kind}"] = (homes[feature] == kind).astype(int)

    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X