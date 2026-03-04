PRD v1 — Sepsis AI Decision Support System
1. Project Title

Hybrid AI System for Early Sepsis Screening using ML, LSTM, and Retrieval-Augmented Generative AI

2. Problem Statement

Sepsis is a life-threatening condition caused by the body's extreme response to infection.
Early detection significantly improves survival rates, but diagnosing sepsis early is difficult because:

symptoms are heterogeneous

physiological signals evolve over time

existing clinical scores (SOFA, qSOFA) have limited predictive power

Traditional machine learning models can predict risk but lack interpretability.
Generative AI models can provide reasoning but require structured medical evidence.

This project aims to build a hybrid AI system combining predictive models and generative reasoning for early sepsis screening.

3. Project Objectives

The system should:

Predict sepsis risk using machine learning models

Model temporal patient data using LSTM

Retrieve similar historical patient cases using RAG

Generate clinical reasoning explanations using a local LLM

Provide interpretable outputs for clinicians

4. Dataset

Primary dataset:

PhysioNet 2019 Sepsis Challenge Dataset

Dataset properties:

ICU patient records

multivariate time-series data

hourly physiological measurements

sepsis labels

Example features:

Heart rate

Mean arterial pressure

Temperature

Respiratory rate

Lactate

White blood cell count

Creatinine

Oxygen saturation

5. System Components

The system will consist of four main modules.

5.1 Data Processing Module

Responsibilities:

load dataset

clean missing values

normalize features

create patient time windows

Outputs:

ML feature dataset

LSTM time-series dataset

patient state descriptions

5.2 Machine Learning Baseline

Purpose:

Provide baseline prediction performance.

Models to implement:

Logistic Regression

Random Forest

XGBoost (optional)

Output:

Sepsis risk probability
5.3 LSTM Time-Series Model

Purpose:

Capture temporal physiological trends.

Input:

Patient vitals across time window

Example sequence:

HR_t1 HR_t2 HR_t3 ...
MAP_t1 MAP_t2 MAP_t3 ...

Output:

Sepsis risk probability
5.4 Retrieval-Augmented Generation (RAG)

Purpose:

Retrieve similar patient cases for reasoning.

Components:

embedding model

vector database

case retriever

Input:

current patient state

Output:

similar historical patient cases
5.5 LLM Reasoning Module

Local LLM models available:

qwen2.5:7b

deepseek-r1:7b

phi3-mini

Primary reasoning model:

qwen2.5:7b

Input prompt:

patient summary
+
retrieved cases
+
prediction outputs

Output:

Sepsis risk assessment
clinical explanation
6. Final System Output

Example output:

Sepsis Risk: HIGH

ML Prediction: 0.82
LSTM Prediction: 0.79

Reasoning:
Elevated lactate and decreasing MAP indicate
possible tissue hypoperfusion consistent with
early sepsis patterns.

Confidence: 0.81
7. Evaluation Metrics

The system will be evaluated using:

Classification metrics

AUROC

Accuracy

Precision

Recall

F1 score

Interpretability metrics

reasoning coherence

clinician interpretability

8. Project Development Phases
Phase 1 — Dataset pipeline

load dataset

preprocess data

exploratory analysis

Phase 2 — ML baseline

train logistic regression

train random forest

evaluate baseline performance

Phase 3 — LSTM model

build time-series dataset

train LSTM network

evaluate performance

Phase 4 — RAG pipeline

create patient case database

build embeddings

implement retriever

Phase 5 — LLM reasoning

integrate local LLM

generate explanations

combine predictions + reasoning

Phase 6 — Final integrated system

combine ML + LSTM + RAG + LLM

generate full decision output

9. Expected Outcome

The final system should demonstrate:

predictive accuracy comparable to traditional ML models

improved interpretability via generative reasoning

a hybrid architecture combining predictive AI and generative AI
