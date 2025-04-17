import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Loading datasets

# Load temperature dataset
temp_df = pd.read_csv(r"C:\Users\rigna_nwlmn\Downloads\ai case study\ML-Location-Case-Study\data\average_monthly_temperature_by_state_1950-2022.csv")
temp_df.columns = temp_df.columns.str.lower().str.strip()

# Load ZIP w/ State mapping
zip_df = pd.read_csv(r"C:\Users\rigna_nwlmn\Downloads\ai case study\ML-Location-Case-Study\data\us_zip_to_state_full.csv")

# Encode all state names
le = LabelEncoder()
temp_df['state_encoded'] = le.fit_transform(temp_df['state'])

# Average temp per state across all years/months
avg_state_temp_df = temp_df.groupby(['state', 'state_encoded'])['average_temp'].mean().reset_index()

X = avg_state_temp_df[['state_encoded']]
y = avg_state_temp_df['average_temp']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Abbreviation w/ all states name's mapped
state_abbr_to_name = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
}

# Get state name from ZIP code
def get_state_from_zip(zip_code):
    match = zip_df[zip_df['Zipcode'] == int(zip_code)]
    if not match.empty:
        return match.iloc[0]['State']
    return None

# Predict avg temp per state using ZIP code 
def predict_temp_by_zip(zip_code):
    abbr = get_state_from_zip(zip_code)
    if not abbr:
        return f"ZIP code {zip_code} not found in ZIP-to-state dataset."

    state_name = state_abbr_to_name.get(abbr, None)
    if not state_name:
        return f"State abbreviation '{abbr}' not found in mapping."

    try:
        state_encoded = le.transform([state_name])[0]
    except ValueError:
        return f"State '{state_name}' is not in the model training data."

    input_data = [[state_encoded]]
    prediction = model.predict(input_data)[0]
    return f"Predicted average temperature for ZIP code {zip_code} ({state_name}): {prediction:.2f} degrees fahrenheit (based on long-term average)"

# Example usage
print(predict_temp_by_zip(78613))  # Texas zipcode
