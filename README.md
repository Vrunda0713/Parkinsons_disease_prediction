

# 🧠 Parkinson’s Disease Prediction Web App

This project is a web-based Machine Learning application for **predicting Parkinson’s Disease** using audio signal features extracted from user-uploaded voice recordings. It also includes educational pages, hospital resource locators, and an AI-powered chatbot to answer questions related to Parkinson’s disease.

---

## 📌 Table of Contents

- [🧠 Overview](#-overview)  
- [✨ Features](#-features)  
- [🧰 Tech Stack](#-tech-stack)  
- [📁 Project Structure](#-project-structure)  
- [🚀 How to Run](#-how-to-run)  
- [📡 API Endpoints](#-api-endpoints)  
- [🧪 Prediction Logic](#-prediction-logic)  
- [🗨️ Chatbot Q&A](#-chatbot-qa)  


---

## 🧠 Overview

This Flask-based web application helps in the **early detection of Parkinson's Disease** through the analysis of speech patterns. Using an **XGBoost classifier** trained on extracted audio features (MFCCs, Chroma, Spectral Contrast), the app provides predictions along with confidence scores.

The app also contains a basic **chatbot**, **hospital locator**, and **Parkinson’s information** center for better accessibility and support.

---

## ✨ Features

- 🎤 **Voice Analysis for Prediction**  
  Upload an audio sample and receive a Parkinson’s disease risk assessment.

- 🧪 **XGBoost ML Model**  
  Machine learning model trained on pre-extracted features for accurate predictions.

- 📈 **Audio Feature Extraction**  
  Uses MFCCs, Chroma, and Spectral Contrast features via Librosa.

- 💬 **Chatbot Q&A Assistant**  
  Interactive chatbot for answering frequently asked questions about Parkinson's.

- 🏥 **Hospital Locator Page**  
  Static page listing nearby or notable hospitals for Parkinson’s treatment.

- 📚 **Educational Info Page**  
  Overview of symptoms, diagnosis, treatment, and prevention of Parkinson’s.

- 📝 **Questionnaire Interface**  
  A basic interactive survey or questionnaire interface (placeholder).

- 🌐 **Flask Web UI**  
  Clean HTML-based templates (`index.html`, `questionnaire.html`, etc.)

---

## 🧰 Tech Stack

| Tool           | Purpose                                |
|----------------|----------------------------------------|
| Python         | Core programming language              |
| Flask          | Web server backend                     |
| Flask-CORS     | Cross-Origin support for frontend-backend communication |
| Librosa        | Audio processing & feature extraction  |
| NumPy          | Numerical computing                    |
| XGBoost        | Machine Learning classification model  |
| Joblib         | Model serialization                    |
| HTML/CSS       | Frontend template rendering            |

---

## 📁 Project Structure

```bash
parkinsons_prediction_app/
├── app.py                       # Main Flask application
├── templates/                   # HTML templates
│   ├── index.html
│   ├── index1.html
│   ├── questionnaire.html
│   ├── parkinsonsinfo.html
│   └── hospitals.html
├── xgb_parkinsons_model.pkl    # Trained XGBoost model
├── scaler.pkl                  # Preprocessing scaler
├── My_New_Flask_app_zipped/    # Optional zipped model structure
└── README.md                   # Project documentation
```

---

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install flask flask-cors numpy librosa xgboost joblib
   ```

2. **Ensure Required Files**:
   - `xgb_parkinsons_model.pkl` and `scaler.pkl` must be in the project root or as per paths in `app.py`.

3. **Run the App**:
   ```bash
   python app.py
   ```

4. **Open in Browser**:
   Navigate to [http://localhost:5500](http://localhost:5500)

---

## 📡 API Endpoints

| Route              | Method | Description |
|-------------------|--------|-------------|
| `/`               | GET    | Homepage (`index.html`) |
| `/index1`         | GET    | Alternate homepage |
| `/questionnaire`  | GET    | Questionnaire/survey page |
| `/parkinsonsinfo` | GET    | Parkinson's disease info page |
| `/hospitals`      | GET    | Static hospital locator page |
| `/predict`        | POST   | Accepts audio file and returns prediction + confidence |
| `/ask`            | POST   | Accepts user question and returns chatbot response |

---

## 🧪 Prediction Logic

- Audio is processed using `librosa` to extract:
  - **MFCCs** (Mel-frequency cepstral coefficients)
  - **Chroma features**
  - **Spectral Contrast**
- Extracted features are:
  - **Padded/trimmed** to match the scaler input shape
  - **Standardized** using a pre-fitted scaler
- Transformed features are passed into a **XGBoost classifier** to generate a probability score.
- Final result is labeled:
  - ✅ **Healthy** if score ≤ 0.5
  - 🟠 **Parkinson’s Detected** if score > 0.5

---

## 🗨️ Chatbot Q&A

Simple rule-based chatbot included at `/ask` endpoint, responds to:
- What is Parkinson's?
- What are the symptoms?
- Is there a cure?
- Who is at risk?
- Can exercise help?
- ...and more.

