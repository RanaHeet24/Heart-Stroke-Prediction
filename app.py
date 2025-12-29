# ------------------------------
# Heart Stroke Prediction UI (Streamlit)
# - KNN model trained on your custom CSV (100 rows)
# - UI exactly as discussed: two sections (Symptoms, Lifestyle & Health)
# - After prediction: shows overall probability % + per-question % contribution
# - Clean medical UI styling + subtle animations
# ------------------------------

import os
import math
import pandas as pd
import numpy as np

import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ==============================
# ---------- CONFIG ------------
# ==============================
st.set_page_config(
    page_title="Heart Stroke Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a modern tech/medical look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    :root {
        --bg-color: #0f172a; /* Slate 900 */
        --card-bg-color: #1e293b; /* Slate 800 */
        --primary-text-color: #f1f5f9; /* Slate 100 */
        --secondary-text-color: #94a3b8; /* Slate 400 */
        --accent-color-1: #22d3ee; /* Cyan 400 */
        --accent-color-2: #f472b6; /* Pink 400 */
        --low-risk-color: #4ade80; /* Green 400 */
        --moderate-risk-color: #facc15; /* Yellow 400 */
        --high-risk-color: #f87171; /* Red 400 */
    }

    body, .st-emotion-cache-1y4p8pa {
        font-family: 'Poppins', sans-serif;
        background: var(--bg-color);
    }

    .main {
        background-color: var(--bg-color);
        color: var(--primary-text-color);
    }
    
    .st-emotion-cache-1y4p8pa { /* Main content area */
        background: linear-gradient(180deg, rgba(15, 23, 42, 0) 0%, #0f172a 150px), 
                    radial-gradient(60% 80% at 50% 0%, #1e293b77 0%, #0f172a 100%);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-text-color) !important;
        font-weight: 700;
    }
    
    .st-caption {
        color: var(--secondary-text-color);
    }

    .med-card { 
        background: var(--card-bg-color);
        border: 1px solid #334155; /* Slate 700 */
        padding: 1.5rem; 
        border-radius: 18px; 
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        height: 100%;
    }
    .med-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.4);
        border-color: var(--accent-color-1);
    }

    /* Custom Radio buttons as toggles */
    div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-start;
    }
    div[role="radiogroup"] > label {
        background-color: var(--card-bg-color);
        border: 1px solid #334155;
        padding: 0.5rem 1.2rem;
        border-radius: 12px;
        margin: 0.2rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    div[role="radiogroup"] > label:hover {
        border-color: var(--accent-color-1);
        background-color: #334155;
    }
    /* Selected radio button */
    div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        background-color: var(--accent-color-1);
        color: var(--bg-color);
        border-color: var(--accent-color-1);
        font-weight: 600;
    }

    .pill { 
        display:inline-block; 
        padding: 0.3rem 0.8rem; 
        border-radius: 999px; 
        background: rgba(34, 211, 238, 0.1); 
        color: var(--accent-color-1); 
        font-weight:600; 
        font-size:0.8rem; 
        border: 1px solid var(--accent-color-1);
    }
    
    .reason { 
        font-size: 0.9rem; 
        color: var(--secondary-text-color);
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0;
    }

    .disclaimer { 
        font-size:0.9rem; 
        color: var(--secondary-text-color);
        border-left: 3px solid var(--moderate-risk-color);
        padding: 1rem;
        background: var(--card-bg-color);
        border-radius: 12px;
    }

    .fade-in { 
        animation: fadeIn 0.6s ease-in-out forwards; 
    }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(10px);} 
        to { opacity: 1; transform: translateY(0);} 
    }

    /* Result styles */
    .final-risk-card {
        text-align: center;
        padding: 2rem;
    }
    .risk-value {
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    .risk-level-LOW { color: var(--low-risk-color); }
    .risk-level-MODERATE { color: var(--moderate-risk-color); }
    .risk-level-HIGH { color: var(--high-risk-color); }

    .analysis-card h4 {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--secondary-text-color);
    }
    .analysis-card .score {
        font-size: 2rem;
        font-weight: 600;
        color: var(--primary-text-color);
    }
    .analysis-card .score-percent {
        font-size: 1.2rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .stButton>button {
        background-color: var(--accent-color-2);
        color: var(--primary-text-color);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.2);
        box-shadow: 0 0 15px var(--accent-color-2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# ------- DATA & MODEL ---------
# ==============================
@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=True)
def train_model_on_selected_columns(df: pd.DataFrame):
    # We will use exactly the UI fields (5 symptoms + 8 lifestyle = 13 features)
    feature_cols = [
        # Symptoms
        "Weakness_One_Side", "Speech_Trouble", "Vision_Problem", "Severe_Headache", "Dizziness_Balance",
        # Lifestyle & Health
        "High_BP", "Diabetes", "Heart_Disease", "Smoke", "Alcohol", "Overweight", "Sleep_Issue", "Family_History",
    ]

    # Encode Yes/No to 1/0 if needed
    df_enc = df.copy()
    for c in feature_cols + ["Stroke"]:
        if df_enc[c].dtype == object:
            df_enc[c] = df_enc[c].map({"Yes":1, "No":0})

    X = df_enc[feature_cols]
    y = df_enc["Stroke"]

    # Balance classes using SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Tune K
    grid = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": list(range(3, 21))}, cv=5, scoring="accuracy")
    grid.fit(X_train_s, y_train)
    best_k = grid.best_params_["n_neighbors"]

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:,1]
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "model": model,
        "acc": acc,
        "report": report,
        "cm": cm,
        "best_k": best_k,
    }

# ==============================
# ------- EXPLAINABLE % --------
# ==============================
# We build a transparent scoring where each Yes contributes a % weight.
# Total weights sum to 100. These are clinical-intuition-inspired, not a substitute for a doctor.
WEIGHTS = {
    # Symptoms (total ~60%)
    "Weakness_One_Side": 15,
    "Speech_Trouble": 12,
    "Vision_Problem": 10,
    "Severe_Headache": 11,
    "Dizziness_Balance": 12,
    # Lifestyle & Health (total ~40%)
    "High_BP": 8,
    "Diabetes": 6,
    "Heart_Disease": 6,
    "Smoke": 5,
    "Alcohol": 3,
    "Overweight": 4,
    "Sleep_Issue": 3,
    "Family_History": 5,
}
assert sum(WEIGHTS.values()) == 100, "Weights must sum to 100"

def explainable_percentages(inputs_dict: dict):
    # inputs_dict: {feature: 0/1}
    contrib = {}
    score = 0
    for k, w in WEIGHTS.items():
        c = w if inputs_dict.get(k, 0) == 1 else 0
        contrib[k] = c
        score += c
    # score is the sum of weights where user answered Yes → out of 100
    return score, contrib

# Friendly display names & reasons (same as spec)
DISPLAY = {
    "Weakness_One_Side": ("Weakness or numbness on one side?", "Classic stroke sign caused by blocked blood flow in the brain."),
    "Speech_Trouble": ("Trouble speaking or understanding people?", "Stroke can affect brain areas for speech and language."),
    "Vision_Problem": ("Sudden blurred or lost vision?", "Stroke in the brain’s vision area can cause eyesight issues."),
    "Severe_Headache": ("Sudden, very strong headache?", "Can be a sign of bleeding in the brain (hemorrhagic stroke)."),
    "Dizziness_Balance": ("Sudden dizziness or loss of balance?", "Stroke often damages the brain’s coordination centers."),
    "High_BP": ("High blood pressure?", "High blood pressure is the main cause of stroke."),
    "Diabetes": ("Diabetes (sugar problem)?", "Diabetes increases blood clotting and damages vessels."),
    "Heart_Disease": ("Any heart disease?", "Heart problems cause irregular blood flow and clots."),
    "Smoke": ("Do you smoke (cigarettes/tobacco)?", "Smoking narrows arteries and thickens blood."),
    "Alcohol": ("Do you drink alcohol?", "Excess alcohol raises blood pressure and weakens the heart."),
    "Overweight": ("Overweight (told by a doctor)?", "Extra weight leads to BP, diabetes, and cholesterol problems."),
    "Sleep_Issue": ("Trouble sleeping often?", "Sleep problems increase BP and stress on the heart."),
    "Family_History": ("Family history of stroke/heart attack?", "Family history means higher genetic risk of stroke."),
}

# Suggested advice snippets per factor (very short, non-diagnostic)
ADVICE = {
    "High_BP": "Monitor BP regularly; reduce salt; discuss medication with a doctor.",
    "Diabetes": "Keep sugar in range; follow diet plan; regular HbA1c checks.",
    "Heart_Disease": "Follow cardiology plan; adhere to meds; watch for chest pain/shortness of breath.",
    "Smoke": "Strongly consider quitting; seek cessation support.",
    "Alcohol": "Limit intake; aim for alcohol-free days each week.",
    "Overweight": "Target gradual weight loss with balanced diet + activity.",
    "Sleep_Issue": "Screen for sleep apnea; aim for 7–8 hours quality sleep.",
    "Family_History": "Discuss risk screening with your clinician; focus on modifiable risks.",
    "Weakness_One_Side": "If sudden/new → emergency care immediately (call local emergency number).",
    "Speech_Trouble": "If sudden/new → emergency care immediately.",
    "Vision_Problem": "If sudden/new → emergency care immediately.",
    "Severe_Headache": "If worst-ever/sudden → emergency care immediately.",
    "Dizziness_Balance": "If sudden with other symptoms → emergency care immediately.",
}

# ==============================
# ----------- UI --------------
# ==============================
st.title("❤️ Heart Stroke Prediction")
st.caption("AI-powered stroke risk analysis using a K-Nearest Neighbors (KNN) model.")

# Load + Train
csv_path = "stroke_prediction_custom_dataset.csv"  # keep the file in the same folder as this app
if not os.path.exists(csv_path):
    st.warning("CSV not found in app folder. Upload your 'stroke_prediction_custom_dataset.csv' via the sidebar.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_loaded = pd.read_csv(uploaded)
        df_loaded.to_csv(csv_path, index=False)
        st.success("CSV saved. Please rerun the app.")
        st.stop()

df = load_dataset(csv_path)
model_pack = train_model_on_selected_columns(df)

with st.expander("Model Details (KNN)", expanded=False):
    st.markdown(f"""
    <div class="med-card">
        <p>This application uses a <strong>K-Nearest Neighbors (KNN)</strong> machine learning model to predict stroke risk based on your inputs. Here are the current model's performance metrics:</p>
        <ul>
            <li><strong>Model Accuracy:</strong> {model_pack['acc']*100:.1f}%</li>
            <li><strong>Best 'K' Value:</strong> {model_pack['best_k']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --------------- Form ---------------
st.subheader("Biometric & Symptom Inputs")
st.markdown("Provide your information below. All data is processed locally and is not stored.")

inputs = {}

symptoms_keys = ["Weakness_One_Side", "Speech_Trouble", "Vision_Problem", "Severe_Headache", "Dizziness_Balance"]
lifestyle_keys = ["High_BP", "Diabetes", "Heart_Disease", "Smoke", "Alcohol", "Overweight", "Sleep_Issue", "Family_History"]

# Use columns for a more compact layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h5><span style='color: var(--accent-color-2);'>🚨</span> Acute Symptoms</h5>", unsafe_allow_html=True)
    for key in symptoms_keys:
        q, reason = DISPLAY[key]
        inputs[key] = 1 if st.radio(q, ["No", "Yes"], horizontal=True, index=0, key=key) == "Yes" else 0
        st.markdown(f"<div class='reason'><span class='pill'>Reason</span> {reason}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0; border-color: #33415566;'>", unsafe_allow_html=True)


with col2:
    st.markdown("<h5><span style='color: var(--accent-color-1);'>🧬</span> Lifestyle & Chronic Factors</h5>", unsafe_allow_html=True)
    for key in lifestyle_keys:
        q, reason = DISPLAY[key]
        inputs[key] = 1 if st.radio(q, ["No", "Yes"], horizontal=True, index=0, key=key) == "Yes" else 0
        st.markdown(f"<div class='reason'><span class='pill'>Reason</span> {reason}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0; border-color: #33415566;'>", unsafe_allow_html=True)


# --------------- Predict Button ---------------
if st.button("⚡ Analyze Risk Profile", use_container_width=True):
    with st.spinner("Running neural analysis..."):
        # Prepare feature vector
        x_vec = np.array([[inputs[k] for k in model_pack["feature_cols"]]])
        x_scaled = model_pack["scaler"].transform(x_vec)

        # Model probability (0..1)
        proba = float(model_pack["model"].predict_proba(x_scaled)[0,1])
        
        # New scoring based on user request
        symptom_score = sum(inputs[k] for k in symptoms_keys)
        lifestyle_score = sum(inputs[k] for k in lifestyle_keys)
        
        symptom_max_score = len(symptoms_keys)
        lifestyle_max_score = len(lifestyle_keys)

        symptom_contrib = (symptom_score / symptom_max_score) * 25 if symptom_max_score > 0 else 0
        lifestyle_contrib = (lifestyle_score / lifestyle_max_score) * 20 if lifestyle_max_score > 0 else 0
        
        # Base risk from model, scaled to be a component of the total risk
        model_base_risk = proba * 50
        
        final_risk = model_base_risk + symptom_contrib + lifestyle_contrib
        final_risk = min(final_risk, 99.0) # Cap risk

        risk_level = "LOW"
        if final_risk > 65:
            risk_level = "HIGH"
        elif final_risk > 35:
            risk_level = "MODERATE"

    st.markdown("---")
    st.header("Analysis Complete: Risk Profile")

    # --- Main Result Display ---
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(f"""
        <div class="med-card final-risk-card fade-in">
            <p class="risk-level-{risk_level}" style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0;">{risk_level} RISK</p>
            <p class="risk-value risk-level-{risk_level}">{final_risk:.1f}%</p>
            <p style="color: var(--secondary-text-color); font-size: 0.9rem;">Calculated Stroke Risk Probability</p>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="med-card fade-in" style="animation-delay: 0.2s; height: 100%;">
            <h4>Key Insights</h4>
            <p style="font-size: 0.9rem;">Your risk profile is primarily driven by {'a combination of lifestyle factors and reported symptoms' if lifestyle_score > 0 and symptom_score > 0 else 'reported symptoms' if symptom_score > 0 else 'lifestyle and chronic factors' if lifestyle_score > 0 else 'the baseline model prediction'}.</p>
            <p style="font-size: 0.9rem;">The AI model assigned a base probability of <strong>{proba*100:.1f}%</strong>, which was then adjusted based on your specific inputs.</p>
            {'<p style="color: var(--high-risk-color); font-size: 0.9rem;"><strong>Immediate attention to acute symptoms is strongly advised.</strong></p>' if symptom_score > 2 else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Detailed Breakdown ---
    st.subheader("Risk Component Analysis")
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    with analysis_col1:
        st.markdown(f"""
        <div class="med-card analysis-card fade-in" style="animation-delay: 0.4s;">
            <h4><span style='color: var(--accent-color-1);'>🤖</span> AI Model Core</h4>
            <p class="score">{model_base_risk:.1f}%</p>
            <p class="score-percent risk-level-{'HIGH' if proba > 0.5 else 'LOW'}">({proba*100:.1f}% raw prob.)</p>
            <p style="color: var(--secondary-text-color);">Base risk from the trained KNN model.</p>
        </div>
        """, unsafe_allow_html=True)
    with analysis_col2:
        st.markdown(f"""
        <div class="med-card analysis-card fade-in" style="animation-delay: 0.6s;">
            <h4><span style='color: var(--accent-color-2);'>🚨</span> Symptom Impact</h4>
            <p class="score">+{symptom_contrib:.1f}%</p>
            <p class="score-percent">{symptom_score}/{symptom_max_score} answered 'Yes'</p>
            <p style="color: var(--secondary-text-color);">Contribution from acute symptom inputs.</p>
        </div>
        """, unsafe_allow_html=True)
    with analysis_col3:
        st.markdown(f"""
        <div class="med-card analysis-card fade-in" style="animation-delay: 0.8s;">
            <h4><span style='color: var(--accent-color-1);'>🧬</span> Lifestyle Factors</h4>
            <p class="score">+{lifestyle_contrib:.1f}%</p>
            <p class="score-percent">{lifestyle_score}/{lifestyle_max_score} answered 'Yes'</p>
            <p style="color: var(--secondary-text-color);">Contribution from chronic/lifestyle factors.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Recommendations & Disclaimer ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Recommendations & Next Steps")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("""
        <div class="med-card fade-in" style="animation-delay: 1s;">
            <h5>Preventive Care</h5>
            <ul>
                <li>Maintain regular health checkups with your primary care physician.</li>
                <li>Actively monitor blood pressure and cholesterol levels.</li>
                <li>Engage in at least 150 minutes of moderate-intensity exercise per week.</li>
                <li>Follow a balanced diet rich in fruits, vegetables, and whole grains.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with rec_col2:
        st.markdown("""
        <div class="disclaimer fade-in" style="animation-delay: 1.2s;">
            <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong><br>
            This AI tool provides a risk assessment only and is not a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice. If you are experiencing any acute symptoms, seek emergency care immediately.
        </div>
        """, unsafe_allow_html=True)



# Footer aesthetic
st.markdown("\n\n")
st.caption("© Heart Stroke Prediction System • Educational use only")
