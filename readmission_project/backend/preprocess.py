import pandas as pd
import numpy as np
filepath='C:\Acadam\materials 6th sem\crypto\ml\readmission_project\data\diabetic_data.csv'
def load_and_clean(filepath):
    
    df = pd.read_csv(filepath)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Drop columns with too many missing values
    df.drop(columns=['weight', 'payer_code', 'medical_specialty'], 
            inplace=True, errors='ignore')
    
    # Simplify target — 1 if readmitted within 30 days else 0
    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )
    
    # Convert age ranges to numbers
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, 
        '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    
    # Drop rows with missing age
    df.dropna(subset=['age'], inplace=True)
    
    return df


def engineer_features(df):
    
    # Make a clean copy to avoid any issues
    df = df.copy()
    
    # Fill missing numeric columns with 0
    numeric_cols = [
        'num_medications', 'time_in_hospital',
        'number_inpatient', 'number_emergency',
        'number_outpatient', 'number_diagnoses', 'age'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Feature 1 — medication change rate
    df['med_change_rate'] = df['num_medications'] / (df['time_in_hospital'] + 1)
    
    # Feature 2 — total prior visits
    df['total_prior_visits'] = (
        df['number_inpatient'] + 
        df['number_emergency'] + 
        df['number_outpatient']
    )
    
    # Feature 3 — elderly flag
    df['is_elderly'] = (df['age'] >= 65).astype(int)
    
    # Feature 4 — multiple diagnoses flag
    df['multiple_diagnoses'] = (df['number_diagnoses'] > 5).astype(int)
    
    # Verify all engineered columns exist
    print("Columns after engineering:", list(df.columns))
    
    # Select final features
    features = [
        'age', 'time_in_hospital', 'num_medications',
        'number_inpatient', 'number_emergency',
        'med_change_rate', 'total_prior_visits',
        'is_elderly', 'multiple_diagnoses'
    ]
    
    X = df[features]
    y = df['readmitted']
    
    return X, y, features 