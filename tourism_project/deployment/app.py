import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model

# replace with your repoid
model_path = hf_hub_download(repo_id="ranjithkumarsundaramoorthy/tourism-project", filename="tourism_project_model_v1.joblib")

model = joblib.load(model_path)

# Streamlit UI for tourism package purchase prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting
based on customer details.
Please enter the customer details below to get a prediction.
""")

# User input fields for all columns except CustomerID and ProdTaken
age = st.number_input("Age", min_value=18, max_value=61, value=18, step=1)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=127.0, value=10.0, step=0.1)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business","Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=1, step=1)
preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
numberoftrips = st.number_input("Number of Trips Annually", min_value=0.0, max_value=22.0, value=2.0, step=1.0)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=3.0, value=0.0, step=1.0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Senior Executive"]) # Added more options based on common sense
monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0, step=1.0)
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
numberoffollowups = st.number_input("Number of Followups", min_value=0.0, max_value=6.0, value=3.0, step=1.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'ProductPitched': productpitched,
    'NumberOfFollowups': numberoffollowups,
}])


if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]
    result = "Customer Will Purchase" if prediction == 1 else "Customer Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
