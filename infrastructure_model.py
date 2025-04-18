# infrastructure_model.py

import pandas as pd
import joblib
import numpy as np
from datetime import datetime

MODEL_PATH = "infrastructureModel.pkl"
ZIP_FEATURES_PATH = "uszips_2.csv"

def predict_infa_rate_by_zip(zip_code_input):
    model = joblib.load(MODEL_PATH)
    zip_data = pd.read_csv(ZIP_FEATURES_PATH)

    try:
        zip_code_input = int(zip_code_input)
    except ValueError:
        return None

    if zip_code_input not in zip_data['zip'].values:
        return None
    
    match = zip_data[zip_data['zip'] == zip_code_input]
    lat = match['lat'].values[0]
    long = match['lng'].values[0]
    columns = [
    'Severity', 
    'Start_Lat', 
    'Start_Lng', 
    'Distance(mi)', 
    'DelayFromTypicalTraffic(mins)']

    row = pd.Series({col: np.nan for col in columns})
    row['Start_Lat'] = lat
    row['Start_Lng'] = long
    row['Year'] = datetime.now().year
    row['Month'] = datetime.now().month

    return model.predict(pd.DataFrame([row]))[0]