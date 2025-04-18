# housing_model.py

import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "house_model.pkl"
ZIP_FEATURES_PATH = "zip_features_general.csv"

def predict_price_by_zip(zip_code_input):
    model = joblib.load(MODEL_PATH)
    zip_data = pd.read_csv(ZIP_FEATURES_PATH)

    try:
        zip_code_input = int(zip_code_input)
    except ValueError:
        return None

    if zip_code_input not in zip_data['zip_code'].values:
        return None

    row = zip_data[zip_data['zip_code'] == zip_code_input].iloc[0]

    input_df = pd.DataFrame([{
        'lat': row['lat'],
        'lng': row['lng'],
        'population': row['population'],
        'density': row['density']
    }])

    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)  
    return prediction
