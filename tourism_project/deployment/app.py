import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Set page configuration
st.set_page_config(layout="wide")

# Download and load the model
MODEL_REPO_ID = "yashghude/tourism-purchase-prediction-model"
MODEL_FILENAME = "best_tourism_prediction_model_v1.joblib"

try:
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure the model is uploaded to Hugging Face.")
    st.stop() # Stop the app if model can't be loaded

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the new Wellness Tourism Package.
_Fill in the customer details below to get a prediction._
""")

st.subheader("Customer Details")

# User input fields based on the dataset description
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=90, value=30, step=1)
    typeofcontact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business', 'Unemployed'])
    gender = st.selectbox("Gender", ['Male', 'Female'])

with col2:
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1, step=1)
    preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5, step=1)
    numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0, step=1)

with col3:
    monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=25000.0, step=100.0)
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'Aspirational'])
    numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2, step=1)
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=15.0, step=0.5)


# Assemble input into DataFrame, ensuring column names match original features for the pipeline
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'MonthlyIncome': monthlyincome,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'ProductPitched': productpitched,
    'NumberOfFollowups': numberoffollowups,
    'DurationOfPitch': durationofpitch
}])

if st.button("Predict Purchase Likelihood"):  
    if model:
        # Predict probabilities
        prediction_proba = model.predict_proba(input_data)[:, 1]
        
        # Use the same classification threshold as in training
        classification_threshold = 0.45
        prediction = (prediction_proba >= classification_threshold).astype(int)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"The model predicts: **Customer is likely to purchase** (Probability: {prediction_proba[0]:.2f})")
        else:
            st.info(f"The model predicts: **Customer is unlikely to purchase** (Probability: {prediction_proba[0]:.2f})")
        st.write("Probability of purchase:", f"{prediction_proba[0]:.2f}")
    else:
        st.warning("Model not loaded. Cannot make predictions.")
