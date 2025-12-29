# ❤️ Heart Stroke Prediction System (KNN + Streamlit)

### 🩺 AI-based stroke risk assessment with explainable insights  
A healthcare-focused machine learning project that predicts the **risk of heart stroke** based on user symptoms, lifestyle habits, and medical history.  
The system combines a tuned **K-Nearest Neighbors (KNN)** model with a **Streamlit-based medical UI** for intuitive and transparent decision support.

---

## 🏷️ Project Highlights

| Feature | Details |
|---------|----------|
| 🤖 Machine Learning | KNN Classifier + **GridSearchCV** for best K-value |
| ⚖️ Data Balancing | **SMOTE** applied to handle class imbalance |
| 🧠 Explainability | Risk score breakdown: Model + Symptoms + Lifestyle |
| 🏥 Medical UI | Custom CSS, neumorphism cards, & animated layout |
| 🔍 Transparency | Shows accuracy, tuned parameters, & risk insights |
| 🌐 Deploy Ready | Compatible with Streamlit Cloud / Heroku / local |

---

## ✨ UI Preview

> 💡 Interactive form with toggle-based symptoms, progress cards & per-factor explanations  
*(Add a screenshot here later for better visual impact)*

```
streamlit run app.py
```

---

## ⚙️ End-to-End Workflow

```
📌 Load & preprocess dataset
📌 Encode categorical features (Yes/No → 1/0)
📌 Balance dataset with SMOTE
📌 Standardize features with StandardScaler
📌 Optimize K via GridSearchCV (range: 3–20)
📌 Train final KNN model
📌 Predict + Evaluate (Accuracy & Classification Report)
```

---

## 📊 Model Performance

| Metric | Status |
|--------|---------|
| **Accuracy** | **XX%** *(Update after training)* |
| **Best K Value** | Auto-selected via GridSearchCV |
| **Evaluation** | Classification Report + Confusion Matrix |
| **Dataset** | `stroke_prediction_custom_dataset.csv` (custom) |

To recalculate accuracy:
```bash
python model_training.py
```

---

## 📂 Folder Structure

```
📦 Heart-Stroke-Prediction
│
├── app.py                           # Streamlit UI (main application)
├── model_training.py                # ML model training & evaluation
├── stroke_prediction_custom_dataset.csv
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Application
```bash
streamlit run app.py
```

---

## 🎛️ Feature Inputs (User Interface)

### 🔶 Acute / Symptom-Based
- One-sided weakness / numbness
- Speech difficulty or slurred speech
- Sudden vision problems
- Severe headache
- Dizziness or balance issues

### 🔷 Lifestyle & Chronic Factors
- High blood pressure
- Diabetes
- Heart disease history
- Smoking / Alcohol use
- Overweight / Obesity
- Sleep issues
- Family medical history

---

## 🧮 Explainable Risk Calculation

| Component | Impact on Score |
|----------:|-----------------|
| 🤖 Model Probability (KNN) | **50%** of final score |
| 🚨 Acute Symptoms | **Up to 25%** |
| 🧬 Lifestyle Factors | **Up to 20%** |
| 🧾 System Cap | Score capped at **99%** to avoid misinterpretation |

> 🧠 This is a **warning system**, not a diagnostic tool.

---

## ⚠️ Medical Disclaimer

This project is intended **for educational and research purposes only**.  
It does **not** provide medical diagnosis or treatment guidance.  
If experiencing symptoms, please seek immediate professional medical support.

---

## 👤 Author

**Rana Heet**  
📌 GitHub: https://github.com/RanaHeet24  
🏗️ Project Domain: Healthcare • ML • Explainable AI  
💡 Interests: Model Tuning • Practical ML • Deployment

---

## ⭐ Contribute & Support

If this project helped you or inspired you:

👉 **Star the repository** ⭐  
👉 **Share with others**  
👉 **Connect for collaboration**

```
git add .
git commit -m "Updated professional README"
git push
```

---

## 📌 Future Enhancements (Roadmap)

- 🔥 Streamlit Cloud deployment
- 🌐 REST API using Flask/FastAPI
- 📊 Dashboard for trend analysis
- 🏥 Medical-grade documentation (FHIR/HL7 format)
- 🧠 Model explainability via SHAP/XAI

---

### 🎉 Thank you for visiting the repository!
Feel free to explore, fork, and contribute 🚀  
