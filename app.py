import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "customer_insights_model.pkl")
CSV_FILE = os.path.join(BASE_DIR, "data.csv")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

def initialize_system():
    """Initialize the system by ensuring model and data exist"""
    if not os.path.exists(CSV_FILE):
        sample_data = pd.DataFrame({
            'Age': np.random.randint(18, 70, 100),
            'Annual Income': np.random.randint(30000, 150000, 100),
            'Spending Score': np.random.randint(1, 100, 100)
        })
        sample_data.to_csv(CSV_FILE, index=False)
    
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        data = pd.read_csv(CSV_FILE)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        model = KMeans(n_clusters=4, random_state=42)
        model.fit(data_scaled)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

@st.cache_resource
def load_model():
    """Load the model and scaler"""
    try:
        initialize_system()
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def create_visualization(df):
    """Create and return visualization figure"""
    fig, ax = plt.subplots()
    ax.hist(df["Spending Score"], bins=10, alpha=0.7, label="Spending Score")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig

def main():
    st.title("Customer Insights System")
    st.write("Analyze customer behavior and predict insights using machine learning.")

    # Load model
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Failed to load model. Please contact support.")
        return

    # User input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=1000, max_value=200000, value=50000)
    spending_score = st.slider("Spending Score", 1, 100, 50)

    if st.button("Analyze Customer"):
        try:
            # Preprocess input data
            input_data = np.array([[age, income, spending_score]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted Customer Segment: {prediction}")

            # Update data file
            new_data = pd.DataFrame([[age, income, spending_score]], 
                                  columns=["Age", "Annual Income", "Spending Score"])
            new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)

            # Load and display data
            df = pd.read_csv(CSV_FILE)
            st.subheader("Recent Customer Data")
            st.write(df.tail(10))

            # Show distribution
            st.subheader("Spending Score Distribution")
            st.pyplot(create_visualization(df))

        except Exception as e:
            st.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()