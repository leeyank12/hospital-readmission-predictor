import pickle
import pandas as pd

def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features


def build_patient_df(age, time_in_hospital, num_medications,
                     number_inpatient, number_emergency, number_diagnoses):
    
    # Feature engineering — same as training
    med_change_rate = num_medications / (time_in_hospital + 1)
    total_prior_visits = number_inpatient + number_emergency
    is_elderly = 1 if age >= 65 else 0
    multiple_diagnoses = 1 if number_diagnoses > 5 else 0
    
    patient = {
        'age': age,
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'number_inpatient': number_inpatient,
        'number_emergency': number_emergency,
        'med_change_rate': med_change_rate,
        'total_prior_visits': total_prior_visits,
        'is_elderly': is_elderly,
        'multiple_diagnoses': multiple_diagnoses
    }
    
    return pd.DataFrame([patient])


def predict_risk(patient_df, model):
    risk = model.predict_proba(patient_df)[0][1]
    return round(risk * 100, 1)


def what_if(patient_df, model, intervention):
    modified = patient_df.copy()
    
    interventions = {
        'discharge_counselling': {
            'med_change_rate': max(0, float(patient_df['med_change_rate']) - 0.5)
        },
        'follow_up_call': {
            'number_emergency': max(0, int(patient_df['number_emergency']) - 1),
            'total_prior_visits': max(0, int(patient_df['total_prior_visits']) - 1)
        },
        'extend_stay': {
            'time_in_hospital': int(patient_df['time_in_hospital']) + 3
        }
    }
    
    if intervention in interventions:
        for col, val in interventions[intervention].items():
            modified[col] = val
    
    new_risk = round(model.predict_proba(modified)[0][1] * 100, 1)
    return new_risk