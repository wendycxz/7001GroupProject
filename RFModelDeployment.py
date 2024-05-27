import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn

# Display version information for debugging
st.write(f"Scikit-learn version: {sklearn.__version__}")

# Load the trained model
try:
    rf = joblib.load('random_forest_model.joblib')
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

#Load the svaed scaler
scaler=joblib.load('scaler.joblib')

# Title of the Streamlit app
st.title('Will this customer sign up for upcoming marketing campaign?')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Function to get user input from sidebar
def user_input_features():
    with st.form(key='input_form'):
        Age = st.slider('Age', min_value=18, max_value=100, value=30)
        Years_joining_as_customer = st.slider('Years as Customer', min_value=0, max_value=50, value=5)
        Recency = st.slider('Recency', min_value=0, max_value=100, value=10)
        Income = st.number_input('Income', min_value=0, max_value=1000000, value=50000, step=1000)
        KidHome = st.slider('Kid Home', min_value=0, max_value=10, value=1)
        TeenHome = st.slider('Teen Home', min_value=0, max_value=10, value=1)
        Complaints = st.slider('Complaints', min_value=0, max_value=100, value=5)
        Cmp1 = st.slider('Campaign 1', min_value=0, max_value=1, value=0)
        Cmp2 = st.slider('Campaign 2', min_value=0, max_value=1, value=0)
        Cmp3 = st.slider('Campaign 3', min_value=0, max_value=1, value=0)
        Cmp4 = st.slider('Campaign 4', min_value=0, max_value=1, value=0)
        Cmp5 = st.slider('Campaign 5', min_value=0, max_value=1, value=0)
        Education = st.selectbox('Education', ['Graduation', 'PhD', 'Master', 'Others'])
        Marital_Status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Together', 'Widow', 'Others'])
        WineSales = st.number_input('Wine Sales', min_value=0, max_value=10000, value=100, step=10)
        FruitSales = st.number_input('Fruit Sales', min_value=0, max_value=10000, value=50, step=10)
        MeatSales = st.number_input('Meat Sales', min_value=0, max_value=10000, value=100, step=10)
        FishSales = st.number_input('Fish Sales', min_value=0, max_value=10000, value=50, step=10)
        SweetSales = st.number_input('Sweet Sales', min_value=0, max_value=10000, value=50, step=10)
        GoldSales = st.number_input('Gold Sales', min_value=0, max_value=10000, value=50, step=10)
        CatalogPurchases = st.slider('Catalog Purchases', min_value=0, max_value=100, value=5)
        DealPurchases = st.slider('Deal Purchases', min_value=0, max_value=100, value=5)
        StorePurchases = st.slider('Store Purchases', min_value=0, max_value=100, value=10)
        WebPurchases = st.slider('Web Purchases', min_value=0, max_value=100, value=10)
        WebVisitsMonth = st.slider('Web Visits per Month', min_value=0, max_value=100, value=5)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
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
    return None

# Get user input
input_df = user_input_features()

if input_df is not None:
    # Ensure correct feature count
    expected_features = [
        'Age', 'Years_joining_as_customer', 'Recency', 'Income', 'KidHome', 'TeenHome', 'Complaints',
        'Cmp1', 'Cmp2', 'Cmp3', 'Cmp4', 'Cmp5', 'WineSales', 'FruitSales', 'MeatSales', 'FishSales',
        'SweetSales', 'GoldSales', 'CatalogPurchases', 'DealPurchases', 'StorePurchases', 'WebPurchases', 'WebVisitsMonth',
        'Education_Graduation', 'Education_PhD', 'Education_Master', 'Education_Others',
        'Marital_Status_Single', 'Marital_Status_Together', 'Marital_Status_Married', 'Marital_Status_Divorced',
        'Marital_Status_Widow', 'Marital_Status_Others'
    ]

    input_df = input_df[expected_features]  # Ensure the order and count of features

    # Apply the scaler
    input_df_scaled = scaler.transform(input_df)

    # Display user input
    st.subheader('User Input Parameters')
    for feature, value in input_df.iloc[0].items():
        st.write(f"{feature}: {value}")

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
