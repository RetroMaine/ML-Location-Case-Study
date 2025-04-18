# crime_model.py

import os
import pandas as pd
import joblib
import numpy as np

# Paths to the trained XGB model, its preprocessor, and the ZIP→state lookup
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, "crime_model.pkl")
PREPROCESSOR_PATH = os.path.join(SCRIPT_DIR, "crime_preprocessor.pkl")
ZIP_LOOKUP_PATH = os.path.join(SCRIPT_DIR, "uszips.csv")

# Load the ZIP→state lookup once
_zips_df = (
    pd.read_csv(ZIP_LOOKUP_PATH, usecols=["zip", "state_id", "population"])
      .rename(columns={"zip": "zip_code", "state_id": "state_abbr"})
)

def predict_crime_rate_by_zip(zip_code_input):
    """
    Returns the predicted violent-crime rate per 1,000 residents for a given ZIP code,
    or None if the ZIP is invalid or no data is available.
    """
    # coerce to int
    try:
        zip_int = int(zip_code_input)
    except (ValueError, TypeError):
        return None

    # look up ZIP
    row = _zips_df[_zips_df["zip_code"] == zip_int]
    if row.empty:
        return None

    # load model + preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError:
        # model artifacts missing
        return None

    # build a single-row DataFrame
    input_df = pd.DataFrame([{
        "state_abbr": row.iloc[0]["state_abbr"],
        "population": row.iloc[0]["population"]
    }])

    # preprocess & predict
    X_proc = preprocessor.transform(input_df)
    rate = model.predict(X_proc)[0]

    # ensure a float
    try:
        return float(rate)
    except:
        return None
