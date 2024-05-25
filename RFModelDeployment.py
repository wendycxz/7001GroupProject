import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn  # Make sure to import sklearn to verify its version

# Display version information for debugging
st.write(f"Scikit-learn version: {sklearn.__version__}")

# Load the trained model
try:
    rf = joblib.load('random_forest_model.joblib')
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Title of the Streamlit app
st.title('Random Forest Classifier')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Function to get user input from sidebar
def user_input_features():
    Age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    Years_joining_as_customer = st.sidebar.number_input('Years as Customer', min_value=0, max_value=50, value=5)
    Recency = st.sidebar.number_input('Recency', min_value=0, max_value=100, value=10)
    Income = st.sidebar.number_input('Income', min_value=0, max_value=1000000, value=50000)
    KidHome = st.sidebar.number_input('Kid Home', min_value=0, max_value=10, value=1)
    TeenHome = st.sidebar.number_input('Teen Home', min_value=0, max_value=10, value=1)
    Complaints = st.sidebar.number_input('Complaints', min_value=0, max_value=100, value=5)
    Cmp1 = st.sidebar.number_input('Campaign 1', min_value=0, max_value=1, value=0)
    Cmp2 = st.sidebar.number_input('Campaign 2', min_value=0, max_value=1, value=0)
    Cmp3 = st.sidebar.number_input('Campaign 3', min_value=0, max_value=1, value=0)
    Cmp4 = st.sidebar.number_input('Campaign 4', min_value=0, max_value=1, value=0)
    Cmp5 = st.sidebar.number_input('Campaign 5', min_value=0, max_value=1, value=0)
    Education = st.sidebar.selectbox('Education', ['Graduation', 'PhD', 'Master', 'Others'])
    Marital_Status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Together', 'Widow', 'Others'])
    WineSales = st.sidebar.number_input('Wine Sales', min_value=0, max_value=10000, value=100)
    FruitSales = st.sidebar.number_input('Fruit Sales', min_value=0, max_value=10000, value=50)
    MeatSales = st.sidebar.number_input('Meat Sales', min_value=0, max_value=10000, value=100)
    FishSales = st.sidebar.number_input('Fish Sales', min_value=0, max_value=10000, value=50)
    SweetSales = st.sidebar.number_input('Sweet Sales', min_value=0, max_value=10000, value=50)
    GoldSales = st.sidebar.number_input('Gold Sales', min_value=0, max_value=10000, value=50)
    CatalogPurchases = st.sidebar.number_input('Catalog Purchases', min_value=0, max_value=100, value=5)
    DealPurchases = st.sidebar.number_input('Deal Purchases', min_value=0, max_value=100, value=5)
    StorePurchases = st.sidebar.number_input('Store Purchases', min_value=0, max_value=100, value=10)
    WebPurchases = st.sidebar.number_input('Web Purchases', min_value=0, max_value=100, value=10)
    WebVisitsMonth = st.sidebar.number_input('Web Visits per Month', min_value=0, max_value=100, value=5)

    data = {
        'Age': Age,
        'Years_joining_as_customer': Years_joining_as_customer,
        'Recency': Recency,
        'Income': Income,
        'KidHome': KidHome,
        'TeenHome': TeenHome,
        'Complaints': Complaints,
        'Cmp1': Cmp1,
        'Cmp2': Cmp2,
        'Cmp3': Cmp3,
        'Cmp4': Cmp4,
        'Cmp5': Cmp5,
        'WineSales': WineSales,
        'FruitSales': FruitSales,
        'MeatSales': MeatSales,
        'FishSales': FishSales,
        'SweetSales': SweetSales,
        'GoldSales': GoldSales,
        'CatalogPurchases': CatalogPurchases,
        'DealPurchases': DealPurchases,
        'StorePurchases': StorePurchases,
        'WebPurchases': WebPurchases,
        'WebVisitsMonth': WebVisitsMonth,
        'Education_Graduation': 1 if Education == 'Graduation' else 0,
        'Education_PhD': 1 if Education == 'PhD' else 0,
        'Education_Master': 1 if Education == 'Master' else 0,
        'Education_Others': 1 if Education == 'Others' else 0,
        'Marital_Status_Single': 1 if Marital_Status == 'Single' else 0,
        'Marital_Status_Together': 1 if Marital_Status == 'Together' else 0,
        'Marital_Status_Married': 1 if Marital_Status == 'Married' else 0,
        'Marital_Status_Divorced': 1 if Marital_Status == 'Divorced' else 0,
        'Marital_Status_Widow': 1 if Marital_Status == 'Widow' else 0,
        'Marital_Status_Others': 1 if Marital_Status == 'Others' else 0,
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
try:
    prediction = rf.predict(input_df)
    prediction_proba = rf.predict_proba(input_df)

    # Display prediction and prediction probabilities
    st.subheader('Prediction')
    st.write('Class: ', prediction[0])
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
