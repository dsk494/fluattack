import streamlit as st
import openai
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Flu Risk Predictor", page_icon="🦠", layout="wide")


PROXY_ENDPOINT = "https://nova-litellm-proxy.onrender.com/"
openai.api_key = 'sk-7apBTwf_uvn5jnw5CE2V_Q' 
openai.api_base = PROXY_ENDPOINT  


def get_health_advice(symptoms):
    prompt = f"Based on the following symptoms: {symptoms}, provide advice on flu risk and precautions to take. Give recommendations for whether a person should consult a healthcare provider, and general tips to reduce flu risk."
    
    try:

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
  
        return response.choices[0].message['content'].strip()
    except Exception as e:

        return f"Error fetching health advice: {str(e)}"


def generate_mock_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'fever': np.random.randint(0, 6, n_samples),
        'fatigue': np.random.randint(0, 6, n_samples),
        'cough': np.random.randint(0, 6, n_samples),
        'body_aches': np.random.randint(0, 6, n_samples),
        'headache': np.random.randint(0, 6, n_samples),
        'sleep_hours': np.random.normal(7, 2, n_samples),
        'stress_level': np.random.randint(0, 6, n_samples),
        'sick_contact': np.random.choice([0, 1], n_samples),
        'vaccinated': np.random.choice([0, 1], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'comorbidity_asthma': np.random.choice([0, 1], n_samples),
        'comorbidity_diabetes': np.random.choice([0, 1], n_samples)
    }
    
    features = pd.DataFrame(data)
    

    probability = (0.25 * features['fever'] / 5 +
                  0.2 * features['fatigue'] / 5 +
                  0.15 * features['cough'] / 5 +
                  0.15 * features['body_aches'] / 5 +
                  0.1 * features['headache'] / 5 +
                  0.1 * (1 - features['sleep_hours'] / 12) +
                  0.1 * features['stress_level'] / 5 +
                  0.2 * features['sick_contact'] +
                  -0.1 * features['vaccinated'] +
                  0.05 * (features['age'] / 100) +
                  0.1 * features['comorbidity_asthma'] +
                  0.1 * features['comorbidity_diabetes'])
    
    features['flu_positive'] = (probability > 0.5).astype(int)
    
    return features

def train_model():
    data = generate_mock_data()
    X = data.drop('flu_positive', axis=1)
    y = data['flu_positive']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler


def create_app():
    st.title("College Flu Risk Predictor 🦠")
    st.markdown("""
        <p style="font-size:16px; color:gray;">Assess your flu risk based on symptoms, lifestyle factors, and medical history.</p>
    """, unsafe_allow_html=True)
    

    with st.expander("🩺 **Enter Symptoms**"):
        col1, col2 = st.columns(2)
        with col1:
            fever = st.slider("Fever Severity (0-5)", 0, 5, 0)
            fatigue = st.slider("Fatigue Level (0-5)", 0, 5, 0)
            cough = st.slider("Cough Severity (0-5)", 0, 5, 0)
            body_aches = st.slider("Body Aches (0-5)", 0, 5, 0)
            headache = st.slider("Headache Intensity (0-5)", 0, 5, 0)
        with col2:
            sleep = st.number_input("Hours of Sleep (last 24h)", 0, 24, 7)
            stress = st.slider("Stress Level (0-5)", 0, 5, 0)
            contact = st.checkbox("Close Contact with Sick People", value=False)
            vaccinated = st.checkbox("Vaccinated against Flu", value=False)

    with st.expander("🏥 **Medical History**"):
        age = st.slider("Age (18-80)", 18, 80, 20)
        asthma = st.checkbox("Do you have asthma?", value=False)
        diabetes = st.checkbox("Do you have diabetes?", value=False)

    if st.button("🔮 **Predict Flu Risk**"):

        input_data = pd.DataFrame({
            'fever': [fever],
            'fatigue': [fatigue],
            'cough': [cough],
            'body_aches': [body_aches],
            'headache': [headache],
            'sleep_hours': [sleep],
            'stress_level': [stress],
            'sick_contact': [int(contact)],
            'vaccinated': [int(vaccinated)],
            'age': [age],
            'comorbidity_asthma': [int(asthma)],
            'comorbidity_diabetes': [int(diabetes)]
        })

    
        input_scaled = scaler.transform(input_data)
        
  
        probability = model.predict_proba(input_scaled)[0][1]
        

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Flu Risk"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        st.plotly_chart(fig)
        
    
        if probability < 0.33:
            st.success("Low Risk: Monitor your symptoms")
        elif probability < 0.66:
            st.warning("Moderate Risk: Consider seeing a healthcare provider")
        else:
            st.error("High Risk: Please seek medical attention")


        symptoms = f"Fever: {fever}, Fatigue: {fatigue}, Cough: {cough}, Body Aches: {body_aches}, Headache: {headache}, Sleep Hours: {sleep}, Stress Level: {stress}, Sick Contact: {contact}, Vaccinated: {vaccinated}, Age: {age}, Asthma: {asthma}, Diabetes: {diabetes}"
        
  
        advice = get_health_advice(symptoms)
        
        
        st.subheader("💡 Health Advice from GPT-4:")
        st.write(advice)

if __name__ == "__main__":
    model, scaler = train_model()
    create_app()
