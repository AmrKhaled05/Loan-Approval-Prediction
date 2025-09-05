import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
logistic_regression_model = joblib.load("logistic_regression_model.joblib")
knn_model = joblib.load("knn_model.joblib")
decision_tree_model = joblib.load("decision_tree_model.joblib")
random_forest_model = joblib.load("random_forest_model.joblib")

svm_model = joblib.load("svm_model.joblib")
gb_model = joblib.load("gradient_boosting_model.joblib")

st.title("Loan Approval Prediction")

no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
income_annum = st.number_input('Annual Income', min_value=0, value=0)
loan_amount = st.number_input('Loan Amount', min_value=0, value=0)
loan_term = st.number_input('Loan Term (months)', min_value=1, value=12)
cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900, value=650)
residential_assets_value = st.number_input('Residential Assets Value', min_value=0, value=0)
commercial_assets_value = st.number_input('Commercial Assets Value', min_value=0, value=0)
luxury_assets_value = st.number_input('Luxury Assets Value', min_value=0, value=0)
bank_asset_value = st.number_input('Bank Asset Value', min_value=0, value=0)


education_encoded = 0 if education == 'Graduate' else 1
self_employed_encoded = 0 if self_employed == 'No' else 1

if st.button("Predict Loan Approval"):
    input_data = np.array([[no_of_dependents, education_encoded, self_employed_encoded, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]])
    
    lr_pred = logistic_regression_model.predict(input_data)
    knn_pred = knn_model.predict(input_data)
    dt_pred = decision_tree_model.predict(input_data)
    rf_pred = random_forest_model.predict(input_data)
    svm_pred = svm_model.predict(input_data)
    gb_pred = gb_model.predict(input_data)
    
    label_map = {0: "Rejected", 1: "Approved"}

    predictions = [
        ("Logistic Regression", lr_pred[0]),
        ("KNN", knn_pred[0]),
        ("Decision Tree", dt_pred[0]),
        ("Random Forest", rf_pred[0]),
        ("SVM", svm_pred[0]),
        ("Gradient Boosting", gb_pred[0]),
    ]

    for model_name, pred in predictions:
        result = label_map.get(pred, pred)
        if result == "Rejected":
            st.error(f'{model_name} Prediction: {result}')
        else:
            st.success(f'{model_name} Prediction: {result}')
    