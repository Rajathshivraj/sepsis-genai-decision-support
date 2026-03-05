# 🏥 GenAI Clinical Decision Support System for Early Sepsis Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A fully local, end-to-end Generative AI system for ICU sepsis risk prediction, clinical reasoning, and decision support — running entirely on consumer hardware.**

[Features](#-system-capabilities) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-running-the-system) • [Dashboard](#-dashboard-features) • [Reports](#-clinical-pdf-reports)

</div>

---

## 📌 Project Overview

Sepsis is a life-threatening medical emergency that kills millions of patients worldwide each year. Early detection is the single most critical factor in survival — yet it remains one of the most difficult clinical challenges in the ICU.

This project builds a **fully local, privacy-preserving AI-powered ICU monitoring assistant** that integrates:

- 🔬 **Machine learning** for sepsis risk prediction
- 🧠 **LSTM temporal modeling** for patient deterioration tracking
- 📊 **Explainable AI** for clinician trust and transparency
- 🗂️ **Retrieval-Augmented Generation (RAG)** for similar case lookup
- 🤖 **Local LLM reasoning** via Ollama + LLaMA 3
- 🧬 **Digital twin simulations** for treatment intervention modeling
- 📄 **Automated clinical PDF reports**
- 🖥️ **Interactive Streamlit dashboard**

> ⚡ Everything runs **100% locally** on consumer hardware — no cloud APIs, no patient data leaves your machine.

---

## ✨ System Capabilities

| Category | Capability |
|---|---|
| 🤖 **ML Models** | Logistic Regression, Random Forest, XGBoost |
| 🔁 **Temporal Modeling** | LSTM time-series model |
| ⚖️ **Risk Calibration** | Ensemble risk scoring |
| 📋 **Clinical Scoring** | qSOFA, SOFA scores |
| 🔍 **Explainability** | SHAP feature importance |
| 🗃️ **RAG** | FAISS vector search, ICU case retrieval |
| 🧬 **Digital Twin** | Fluid resuscitation & antibiotic intervention simulation |
| 📈 **Forecasting** | Risk trajectory prediction |
| 🦙 **Generative AI** | Local LLM reasoning (Ollama + LLaMA 3) |
| 📊 **Visualization** | Streamlit + Plotly charts |
| 📄 **Reporting** | Multi-section clinical PDF reports |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        🏥 PATIENT ICU DATA INPUT                     │
│              (PhysioNet PSV Files — Vitals, Labs, Sepsis Labels)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ⚙️  FEATURE ENGINEERING                          │
│        Missing value imputation • Temporal windowing • Scaling       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Logistic    │ │ Random       │ │   XGBoost    │
        │ Regression  │ │ Forest       │ │   Model      │
        └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
               └───────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   🔁 LSTM Temporal   │
                    │      Model           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  ⚖️ RISK ENSEMBLE    │
                    │  Calibrated Scoring  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
        │  📋 qSOFA   │ │ 🔍 SHAP XAI  │ │ 📈 Risk      │
        │  SOFA Score │ │ Explainability│ │ Forecasting  │
        └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
               └───────────────┼────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
        │ 🗃️ FAISS    │ │ 📚 Guideline │ │  🧬 Digital  │
        │ Case RAG    │ │ Retrieval    │ │  Twin Sim    │
        └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
               └───────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  🦙 LOCAL LLM       │
                    │  Clinical Reasoning  │
                    │  (Ollama + LLaMA 3)  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌─────────────────────┐         ┌─────────────────────┐
    │ 🖥️ STREAMLIT        │         │ 📄 CLINICAL PDF     │
    │    DASHBOARD         │         │    REPORT            │
    └─────────────────────┘         └─────────────────────┘
```

---

## 📂 Repository Structure

```
genai-sepsis-cds/
│
├── 📁 src/
│   ├── 📁 preprocessing/          # Data loading, cleaning, feature engineering
│   │   ├── loader.py              # PhysioNet PSV file parser
│   │   ├── imputer.py             # Missing value strategies
│   │   └── feature_engineer.py   # Temporal windowing, scaling
│   │
│   ├── 📁 models/                 # ML model training and inference
│   │   ├── logistic.py            # Logistic Regression wrapper
│   │   ├── random_forest.py       # Random Forest with SHAP support
│   │   ├── xgboost_model.py       # XGBoost classifier
│   │   └── ensemble.py            # Risk score ensembler + calibration
│   │
│   ├── 📁 temporal/               # LSTM time-series modeling
│   │   ├── lstm_model.py          # PyTorch LSTM architecture
│   │   ├── sequence_builder.py    # Patient sequence construction
│   │   └── trainer.py             # LSTM training loop
│   │
│   ├── 📁 rag/                    # Retrieval-Augmented Generation
│   │   ├── faiss_index.py         # FAISS vector store
│   │   ├── embedder.py            # Patient feature embeddings
│   │   └── retriever.py           # Cosine similarity case retrieval
│   │
│   ├── 📁 llm/                    # Local LLM integration
│   │   ├── ollama_client.py       # Ollama API wrapper
│   │   ├── prompt_builder.py      # Clinical context prompt construction
│   │   └── reasoner.py            # Clinical reasoning pipeline
│   │
│   ├── 📁 forecasting/            # Risk trajectory prediction
│   │   ├── forecaster.py          # Multi-step risk projection
│   │   └── trend_analyzer.py      # Deterioration trend detection
│   │
│   ├── 📁 digital_twin/           # Treatment intervention simulation
│   │   ├── twin_engine.py         # Simulation core
│   │   ├── interventions.py       # Fluid, antibiotics, vasopressors
│   │   └── outcome_predictor.py   # Post-intervention risk modeling
│   │
│   ├── 📁 explainability/         # XAI module
│   │   ├── shap_explainer.py      # SHAP value computation
│   │   └── feature_ranker.py      # Feature importance summarizer
│   │
│   ├── 📁 clinical_scores/        # Clinical scoring systems
│   │   ├── qsofa.py               # qSOFA calculator
│   │   └── sofa.py                # SOFA score calculator
│   │
│   ├── 📁 visualization/          # Plotly chart components
│   │   ├── risk_gauge.py          # Risk meter gauge
│   │   ├── trajectory_plot.py     # Risk over time charts
│   │   └── vitals_monitor.py      # Vital signs panels
│   │
│   └── 📁 reporting/              # Clinical PDF generation
│       ├── report_builder.py      # ReportLab PDF constructor
│       └── chart_exporter.py      # Embeds charts into PDF
│
├── 📁 ui/
│   └── app.py                     # 🖥️ Main Streamlit dashboard
│
├── 📁 models/                     # Saved model artifacts (.pkl, .pt)
├── 📁 data/                       # PhysioNet PSV patient files
├── 📁 scripts/                    # Training and evaluation scripts
│   ├── train_models.py
│   ├── build_faiss_index.py
│   └── evaluate.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧩 Module Descriptions

### ⚙️ Preprocessing
Loads raw PhysioNet `.psv` patient files, handles missing physiological measurements using forward-fill and median imputation, computes rolling temporal windows, and normalizes features for downstream models.

---

### 🤖 Models
Three independently trained classifiers — Logistic Regression, Random Forest, and XGBoost — each producing a calibrated sepsis risk probability. An ensemble layer combines their outputs using weighted averaging to produce the final risk score.

---

### 🔁 Temporal Modeling (LSTM)
A PyTorch LSTM processes sequences of hourly ICU measurements to capture deterioration dynamics over time. The model learns from 12-hour patient history windows to produce a sequence-aware risk estimate that complements the snapshot-based ML models.

---

### 🗃️ Retrieval-Augmented Generation (RAG)
Patient feature vectors are embedded into a dense representation and stored in a **FAISS** index built from historical ICU cases. At inference time, the system retrieves the top-K most similar past patients using cosine similarity, enabling clinicians to see how analogous cases evolved and were treated.

---

### 🦙 Local LLM Reasoning
A structured clinical prompt is assembled from the patient's vitals, risk scores, clinical scores, SHAP explanations, and retrieved similar cases. This prompt is passed to a **locally running LLaMA 3 model via Ollama**, which generates a clinical reasoning narrative including potential diagnoses, risk factors, and treatment considerations — entirely on-device.

---

### 📈 Forecasting
The forecasting module projects the patient's sepsis risk trajectory over the next 6–12 hours. It models deterioration trends and flags inflection points where risk is accelerating, providing early warning signals before clinical thresholds are crossed.

---

### 🧬 Digital Twin Simulation
The digital twin engine simulates the patient's physiological response to hypothetical interventions:

- 💧 **Fluid resuscitation** — models hemodynamic response
- 💊 **Antibiotic therapy** — models infection control effect
- 🫀 **Vasopressor administration** — models mean arterial pressure response

Each simulation produces a predicted post-intervention risk trajectory, helping clinicians reason about treatment decisions before acting.

---

### 🔍 Explainability (SHAP)
SHAP (SHapley Additive exPlanations) values are computed for the XGBoost model to identify which features most strongly contributed to the current risk score. Results are presented as ranked bar charts and fed into the LLM prompt for explainable clinical reasoning.

---

### 📋 Clinical Scores
Standard critical care scoring systems are computed directly from patient measurements:
- **qSOFA**: respiratory rate, altered mentation, systolic blood pressure
- **SOFA**: multi-organ dysfunction assessment across six organ systems

---

### 📊 Visualization
All charts are built with **Plotly** and rendered inside the Streamlit dashboard, including animated risk gauges, multi-panel vital sign monitors, trajectory forecasts, SHAP bar charts, and digital twin simulation overlays.

---

### 📄 Reporting
The reporting module uses **ReportLab** to generate a structured multi-section PDF clinical report that can be saved, printed, or attached to a patient record.

---

## 🖥️ Dashboard Features

The Streamlit dashboard provides a single-page clinical decision support interface:

| Panel | Description |
|---|---|
| 🎯 **Risk Gauge** | Animated speedometer showing current sepsis risk (Low / Moderate / High / Critical) |
| 💓 **Vital Sign Monitor** | Real-time display of HR, BP, RR, SpO₂, Temperature, Lactate |
| 📈 **Risk Trajectory** | Time-series chart showing risk evolution and 6-hour forecast |
| 🧬 **Digital Twin** | Side-by-side comparison of baseline vs. post-intervention risk trajectories |
| 🗂️ **Similar ICU Cases** | Top-3 historically similar patients with outcomes and treatment summaries |
| 🦙 **AI Clinical Reasoning** | Full LLM-generated clinical narrative with differential diagnoses |
| 🔍 **Feature Importance** | SHAP bar chart explaining the top drivers of the current risk score |

---

## 📄 Clinical PDF Reports

The system generates a structured, print-ready PDF report for each patient assessment containing:

```
📋 CLINICAL DECISION SUPPORT REPORT
════════════════════════════════════

  1. Patient Summary
     └── Demographics, admission data, current vitals

  2. Sepsis Risk Assessment
     └── Ensemble risk score, risk tier, confidence interval

  3. Clinical Scores
     └── qSOFA score and breakdown
     └── SOFA score across 6 organ systems

  4. Risk Trajectory Forecast
     └── Historical trend chart
     └── 6-hour risk projection

  5. Digital Twin Simulations
     └── Baseline trajectory
     └── Fluid resuscitation scenario
     └── Antibiotic therapy scenario

  6. Similar Historical ICU Cases
     └── Top-3 matches with similarity scores and outcomes

  7. AI Clinical Reasoning
     └── Full LLM-generated clinical narrative
     └── Key risk factors
     └── Suggested considerations

  8. SHAP Feature Importance
     └── Top contributing features with direction and magnitude
```

---

## 📦 Dataset

**PhysioNet Computing in Cardiology Challenge 2019 — Sepsis Prediction**

| Property | Value |
|---|---|
| 👥 Patients | ~40,000 ICU patients |
| ⏱️ Resolution | Hourly time-series measurements |
| 🩺 Features | 40 physiological & lab variables |
| 🏷️ Labels | Binary sepsis onset labels |
| 📁 Format | Pipe-separated values (`.psv`), one file per patient |

**Key features include:** Heart Rate, O2 Sat, Temperature, SBP, MAP, DBP, Resp Rate, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, ICULOS.

> 🔗 Dataset available at: [physionet.org/content/challenge-2019](https://physionet.org/content/challenge-2019/)

---

## 💻 Hardware

Developed and tested on:

| Component | Specification |
|---|---|
| 🎮 **GPU** | NVIDIA RTX 5050 Laptop GPU |
| 🖥️ **VRAM** | 8 GB |
| 🧠 **RAM** | 24 GB |
| 💿 **OS** | WSL Ubuntu (Windows Subsystem for Linux) |
| 🦙 **LLM Runtime** | Ollama (local inference) |

> ✅ Designed to run entirely on consumer-grade hardware — no cloud GPU required.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/genai-sepsis-cds.git
cd genai-sepsis-cds
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
# OR
venv\Scripts\activate          # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull LLaMA 3

```bash
# Install Ollama (Linux / WSL)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLaMA 3 model
ollama pull llama3
```

### 5. Download the PhysioNet dataset

```bash
# Place .psv patient files in:
mkdir -p data/training
# Copy your PhysioNet PSV files into data/training/
```

### 6. Configure environment

```bash
cp .env.example .env
# Edit .env with your paths and settings
```

---

## 🚀 Running the System

### Step 1 — Train the ML models

```bash
python scripts/train_models.py
```

### Step 2 — Build the FAISS RAG index

```bash
python scripts/build_faiss_index.py
```

### Step 3 — Start the Ollama LLM server

```bash
ollama serve
```

### Step 4 — Launch the Streamlit dashboard

```bash
streamlit run ui/app.py
```

Then open your browser at: **`http://localhost:8501`**

### Optional — Run evaluation

```bash
python scripts/evaluate.py --model xgboost --split test
```

---

## 📸 Screenshots

> *(Screenshots will be added after dashboard finalization)*

### 🖥️ Main Dashboard
```
[ Dashboard screenshot placeholder ]
Risk Gauge | Vital Signs | Clinical Scores
```

### 📈 Risk Trajectory Chart
```
[ Risk trajectory chart placeholder ]
Historical risk + 6-hour forecast
```

### 🧬 Digital Twin Simulation
```
[ Digital twin chart placeholder ]
Baseline vs. fluid resuscitation vs. antibiotics
```

### 📄 Clinical PDF Report
```
[ PDF report preview placeholder ]
Multi-section structured clinical report
```

---

## 🌟 Project Highlights

### 🔄 Full End-to-End AI Pipeline
From raw PSV files to clinical decision support — every step is automated, integrated, and reproducible.

### 🤖 Generative AI Clinical Reasoning
Rather than just outputting a risk number, the system generates a full **clinical narrative** explaining *why* the patient is at risk and *what* clinical factors are most concerning — using a locally running LLM.

### 🔍 Explainable AI
SHAP values ensure that every risk prediction is **interpretable**. Clinicians can see exactly which features drove the model's decision, building trust and supporting clinical oversight.

### 🧬 Digital Twin Treatment Simulation
The digital twin allows clinicians to ask **"what if?"** questions before committing to a treatment — simulating the expected physiological response to fluid resuscitation, antibiotics, or vasopressors.

### 🔒 Fully Local — Privacy Preserving
No patient data ever leaves the hospital. No cloud APIs. No external services. The LLM runs locally via Ollama, making this system suitable for **real clinical environments** with strict data governance requirements.

### 📄 Automated Clinical Reporting
The system generates structured, print-ready PDF reports that summarize the full AI assessment, suitable for clinical documentation or handover.

---

## ⚠️ Limitations

> **This system is a research prototype and is NOT approved for clinical use.**

- Trained on the PhysioNet Challenge 2019 dataset which may not generalize to all hospital populations or clinical workflows
- The LLM reasoning is generative and may produce plausible-sounding but clinically inaccurate statements — all outputs must be reviewed by a qualified clinician
- The digital twin simulations are simplified physiological models and do not replace clinical pharmacological expertise
- The system has not undergone clinical validation trials
- Performance may degrade on patient populations significantly different from the training distribution

---

## 🔭 Future Work

| Priority | Enhancement |
|---|---|
| 🔴 High | **Real-time ICU monitoring** via HL7 FHIR / bedside monitor integration |
| 🔴 High | **EHR integration** (Epic, Cerner) for seamless clinical workflow embedding |
| 🟡 Medium | **Multi-hospital federated training** for improved generalization |
| 🟡 Medium | **Treatment optimization** via reinforcement learning |
| 🟡 Medium | **Fine-tuned clinical LLM** on medical literature and sepsis guidelines |
| 🟢 Exploratory | **Prospective clinical trial** for real-world validation |
| 🟢 Exploratory | **Multimodal input** — waveform ECG, chest X-ray integration |
| 🟢 Exploratory | **Automated alert system** with configurable clinical thresholds |

---

## 📚 References

- Singer M, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA 2016.
- Reyna MA, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine 2020.
- Lundberg SM, Lee SI. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Johnson AEW, et al. *MIMIC-III, a freely accessible critical care database.* Scientific Data 2016.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [PhysioNet](https://physionet.org/) for the Challenge 2019 dataset
- [Ollama](https://ollama.com/) for making local LLM inference accessible
- [SHAP](https://shap.readthedocs.io/) for model explainability tooling
- [Streamlit](https://streamlit.io/) for rapid dashboard development
- [FAISS](https://faiss.ai/) (Meta AI Research) for efficient vector search

---

<div align="center">

**Built for research, clinical AI education, and ML engineering showcase.**

*If you find this project useful, please consider giving it a ⭐*

</div>
