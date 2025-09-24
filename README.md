# LoanAnalyzer Pro

A full-stack machine learning web app for loan prediction and data analysis, built with React, FastAPI, and deployed on Vercel.

## 🔗 Live Demo

- [Dashboard](https://data-analysis-eloj.vercel.app/)
- [Loan Predictor](https://data-analysis-eloj.vercel.app/predict)
- [Data Analysis](https://data-analysis-eloj.vercel.app/data)

## 🚀 Features

- 📊 Interactive dashboard for loan data insights
- 🤖 ML-powered loan approval predictor
- 📈 Visual analysis of applicant trends
- 🌐 Deployed on Vercel with backend hosted via Render

## 🛠️ Tech Stack

**Frontend**
- React + Chakra UI + MUI
- Axios for API calls
- CRACO for custom build config

**Backend**
- FastAPI + Scikit-learn
- Pickled ML models (`loan_model.pkl`, `scaler.pkl`, `label_encoders.pkl`)
- RESTful endpoints for prediction and data serving

**Deployment**
- Frontend: Vercel (`npm install --legacy-peer-deps`)
- Backend: Render with `render.yaml`


