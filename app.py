import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

with open("clfxgb_model.pkl", "rb") as model_file:
    clfxgb1 = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

all_mappings = {
    'mechanism_of_injury': {
        'Games': 0, 'Fall from Height': 1, 'Machine Injury': 2, 'Road Traffic Accident': 3,
        'Assault': 4, 'GunShot': 5, 'Others': 6
    },
    'respiration': {
        'Normal': 0, 'Periodic': 1, 'Cheyne Stokes': 2, 'Gasping': 3
    },
    'xray_abnormality': {
        'Normal': 0, 'Listhesis': 1, 'Fracture': 2
    },
    'right_eye_vision': {
        'Absent': 4, 'Projection of light': 3, 'Finger counting 1 - 3 ft': 2, 
        'Finger counting 3 - 6 ft': 1, 'Finger counting > 6 ft': 0
    },
    'left_eye_vision': {
        'Absent': 5, 'Projection of light': 4, 'Finger counting 1 - 3 ft': 2, 
        'Finger counting 3 - 6 ft': 1, 'Finger counting > 6 ft': 0, 'Finger counting < 1 ft': 3
    },
    'associated_injury': {
        'No Injury': 0, 'Other': 1, 'Long Bone': 2, 'Pelvic Injury': 3, 
        'Chest Injury': 4, 'SIG Facial Fracture': 5
    },
    'speech': {
        'Normal': 0, 'Dysphasia': 1, 'Aphasia': 2, 'Cannot Assess': 3
    },
    'sex': {
        'Female': 0, 'Male': 1
    },
    'headache': {
        'No': 0, 'Yes': 1
    },
    'vomiting': {
        'No': 0, 'Yes': 1
    },
    'amnesia': {
        'No': 0, 'Yes': 1
    },
    'csf_rhinorrhea': {
        'No': 0, 'Yes': 1
    },
    'head_injury_type': {
        'Closed': 0, 'Open': 1
    },
    'power_rul': {
        '0 power': 0, '1 power': 1, '2 power': 2, '3 power': 3, '4 power': 4, '5 power': 5
    },
    'power_rll': {
        '0 power': 0, '1 power': 1, '2 power': 2, '3 power': 3, '4 power': 4, '5 power': 5
    },
    'power_lul': {
        '0 power': 0, '1 power': 1, '2 power': 2, '3 power': 3, '4 power': 4, '5 power': 5
    },
    'power_lll': {
        '0 power': 0, '1 power': 1, '2 power': 2, '3 power': 3, '4 power': 4, '5 power': 5
    }
}

st.title("TBI Survival Probability Prediction System")
st.sidebar.header("Enter Feature Values:")

selected_features = [
    "pulse", "age", "gcs_evm", "rotterdam_ct_grade", "time_to_first_doctor",
    "time_in_trauma_center", "intracranial_hem_vol", "speech", "left_eye_vision",
    "right_eye_vision", "vomiting", "respiration", "head_injury_type",
    "mechanism_of_injury", "power_lul", "sex", "power_lll", "csf_rhinorrhea",
    "headache", "power_rll", "xray_abnormality", "associated_injury", "power_rul", "amnesia"
]

categorical_vars = list(all_mappings.keys())
continuous_vars = [var for var in selected_features if var not in categorical_vars]

input_values = {}

for feature in continuous_vars:
    input_values[feature] = st.sidebar.number_input(f"{feature}:", value=0.0)

for feature, mapping in all_mappings.items():
    selected_option = st.sidebar.selectbox(f"{feature}:", options=list(mapping.keys()))
    input_values[feature] = mapping[selected_option]

input_array = np.array([input_values[feature] for feature in selected_features]).reshape(1, -1)

scaled_input = scaler.transform(input_array)

if st.sidebar.button("Predict Outcome"):
    prediction = clfxgb1.predict(scaled_input)[0]
    survival_probability = clfxgb1.predict_proba(scaled_input)[0][1]
    death_probability = 1 - survival_probability

    survival_percentage = round(survival_probability * 100, 2)
    death_percentage = round(death_probability * 100, 2)
    if prediction == 1:
        st.subheader(f"{survival_percentage}% probability of survival")
    else:
        st.subheader(f"{death_percentage}% probability of death")

    explainer = shap.TreeExplainer(clfxgb1)
    shap_values = explainer.shap_values(scaled_input)

    st.subheader("SHAP  Plot (Feature Importance for Prediction)")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, np.array(scaled_input), feature_names=selected_features, show=False)
    st.pyplot(fig)