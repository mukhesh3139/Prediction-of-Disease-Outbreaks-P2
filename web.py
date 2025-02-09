import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 2em;
    }
    .prediction-box {
        padding: 2em;
        border-radius: 10px;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_SCALER_FOLDER = "models_scalers"

models = {
    "Diabetes": os.path.join(MODEL_SCALER_FOLDER, "best_diabetes_model.pkl"),
    "Heart Disease": os.path.join(MODEL_SCALER_FOLDER, "best_heart_model.pkl"),
    "Parkinson's": os.path.join(MODEL_SCALER_FOLDER, "best_parkinsons_model.pkl")
}

scalers = {
    "Diabetes": os.path.join(MODEL_SCALER_FOLDER, "diabetes_scaler.pkl"),
    "Heart Disease": os.path.join(MODEL_SCALER_FOLDER, "heart_scaler.pkl"),
    "Parkinson's": os.path.join(MODEL_SCALER_FOLDER, "parkinsons_scaler.pkl")
}

input_features = {
    "Diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "Heart Disease": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    "Parkinson's": [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE"
    ]
}

def load_model_and_scaler(disease):
    with open(models[disease], 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scalers[disease], 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

def create_feature_input(disease):
    features = {}
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(input_features[disease]):
        with col1 if idx % 2 == 0 else col2:
            if feature == "sex":
                features[feature] = st.selectbox(feature, options=[0, 1], help="0: Female, 1: Male")
            elif feature in ["cp", "restecg", "slope", "thal"]:
                features[feature] = st.selectbox(feature, options=[0, 1, 2, 3])
            elif feature in ["fbs", "exang"]:
                features[feature] = st.selectbox(feature, options=[0, 1])
            else:
                features[feature] = st.number_input(feature, value=0.0, format="%.2f")
    
    return features

def predict_disease(features, disease):
    model, scaler = load_model_and_scaler(disease)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    return prediction, prediction_proba

def create_gauge_chart(probability, threshold=0.5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Probability"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold * 100], 'color': "lightgreen"},
                {'range': [threshold * 100, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_feature_importance_and_summary(features, model, disease):
    st.markdown("### Analysis Summary")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Input Summary")
        df = pd.DataFrame([features])
        st.dataframe(df.T.style.set_caption("Patient Data"))
    
    with col2:
        st.markdown("#### Feature Importance")
        # Get feature importance from the model
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importance_values = np.abs(model.coef_[0])
        else:
            st.warning("Feature importance not available for this model type")
            return
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Importance': importance_values
        })
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Create horizontal bar chart using plotly
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color='royalblue',
                colorscale='Blues',
            )
        ))
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of top features
        st.markdown("#### Top Influential Features")
        top_features = importance_df.tail(3)[::-1]
        
        for idx, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
            st.write(f"{idx+1}. **{feature}**: {importance:.3f}")

def main():
    # Header
    st.title("👨‍⚕️ Disease Prediction System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    selected_disease = st.sidebar.selectbox(
        "Select Disease to Predict",
        list(models.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application uses machine learning models to predict the likelihood of various diseases based on patient data.
    
    ### Instructions
    1. Select a disease from the dropdown
    2. Enter patient information
    3. Click 'Predict' to see results
    """)
    
    # Main content
    st.subheader(f"Predict {selected_disease}")
    st.markdown(f"Please enter the required information to predict {selected_disease.lower()}.")
    
    # Create input form
    with st.form(key=f'{selected_disease}_form'):
        features = create_feature_input(selected_disease)
        submit_button = st.form_submit_button(label='Predict')
    
    # Make prediction when form is submitted
    if submit_button:
        prediction, prediction_proba = predict_disease(features, selected_disease)
        
        # Display results
        st.markdown("### Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"⚠️ High Risk of {selected_disease}")
            else:
                st.success(f"✅ Low Risk of {selected_disease}")
        
        with col2:
            positive_prob = prediction_proba[1]
            fig = create_gauge_chart(positive_prob)
            st.plotly_chart(fig)
        
        # Display feature importance or additional information
        
        model, _ = load_model_and_scaler(selected_disease)
        display_feature_importance_and_summary(features, model, selected_disease)
if __name__ == "__main__":
    main()