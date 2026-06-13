import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt
from backend.predict import load_model, build_patient_df, predict_risk, what_if
from backend.explain import plot_shap_bar, get_lime_explanation

# ── Page config ──
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Header ──
st.title("🏥 Hospital Readmission Predictor")
st.markdown("*Predict 30-day readmission risk with Explainable AI*")
st.divider()

# ── Load model ──
model, features = load_model()

# ── Layout ──
col1, col2 = st.columns([1, 1.2])

# ── Left column: Patient inputs ──
with col1:
    st.subheader("Patient Details")

    age = st.slider("Age", 20, 90, 58)
    time_in_hospital = st.slider("Days in Hospital", 1, 20, 7)
    num_medications = st.slider("Number of Medications", 1, 25, 12)
    number_inpatient = st.slider("Prior Inpatient Visits", 0, 10, 4)
    number_emergency = st.slider("Prior Emergency Visits", 0, 10, 2)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 6)

    st.divider()
    predict_btn = st.button(
        "Predict Readmission Risk",
        use_container_width=True
    )

# ── Right column: Results ──
with col2:
    if predict_btn:

        # Build patient dataframe
        patient_df = build_patient_df(
            age, time_in_hospital, num_medications,
            number_inpatient, number_emergency, number_diagnoses
        )

        # Get risk score
        risk_pct = predict_risk(patient_df, model)

        # ── Risk score display ──
        st.subheader("Readmission Risk Score")
        if risk_pct >= 60:
            st.error(f"🔴 HIGH RISK — {risk_pct}% chance of readmission")
        elif risk_pct >= 35:
            st.warning(f"🟡 MEDIUM RISK — {risk_pct}% chance of readmission")
        else:
            st.success(f"🟢 LOW RISK — {risk_pct}% chance of readmission")

        st.progress(int(risk_pct))
        st.divider()

        # ── SHAP explanation ──
        st.subheader("Why this prediction? (SHAP)")
        fig = plot_shap_bar(model, patient_df, features)
        st.pyplot(fig)
        plt.close()
        st.divider()

        # ── LIME explanation ──
        st.subheader("Patient-level explanation (LIME)")
        try:
            from backend.preprocess import load_and_clean, engineer_features
            df_raw = load_and_clean('data/diabetic_data.csv')
            X_all, _, _ = engineer_features(df_raw)
            lime_exp = get_lime_explanation(model, X_all, patient_df, features)

            for feat, weight in lime_exp:
                color = "🔴" if weight > 0 else "🟢"
                st.write(f"{color} `{feat}` — impact: `{round(weight, 3)}`")
        except Exception as e:
            st.info("LIME explanation unavailable for this prediction.")
        st.divider()

        # ── What-If simulator ──
        st.subheader("What-If Simulator")
        st.caption("How much does each intervention reduce risk?")

        interventions = {
            "Add discharge counselling": "discharge_counselling",
            "Schedule follow-up call": "follow_up_call",
            "Extend hospital stay by 3 days": "extend_stay"
        }

        for label, key in interventions.items():
            new_risk = what_if(patient_df, model, key)
            delta = round(new_risk - risk_pct, 1)
            arrow = "↓" if delta < 0 else "↑"
            delta_color = "normal" if delta < 0 else "inverse"
            st.metric(
                label=label,
                value=f"{new_risk}%",
                delta=f"{arrow} {abs(delta)}%",
                delta_color=delta_color
            )

    else:
        st.info("👈 Fill in patient details and click **Predict** to see results.")
        st.markdown("""
        **This app predicts:**
        - 30-day hospital readmission risk
        - Key factors driving the risk (SHAP)
        - Patient-level explanation (LIME)
        - Impact of clinical interventions (What-If)
        """)