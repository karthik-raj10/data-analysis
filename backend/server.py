from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import io


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Loan Default Prediction API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for ML model and scaler
model = None
scaler = None
feature_columns = None

# Pydantic Models
class LoanApplication(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    applicant_income: float = Field(..., description="Annual income of the applicant")
    coapplicant_income: float = Field(default=0, description="Annual income of the co-applicant")
    loan_amount: float = Field(..., description="Loan amount requested")
    loan_amount_term: int = Field(..., description="Loan term in days")
    credit_history: int = Field(..., description="Credit history (1: good, 0: poor)")
    gender: str = Field(..., description="Gender (Male/Female)")
    married: str = Field(..., description="Marital status (Yes/No)")
    dependents: str = Field(..., description="Number of dependents (0/1/2/3+)")
    education: str = Field(..., description="Education level (Graduate/Not Graduate)")
    self_employed: str = Field(..., description="Self employed (Yes/No)")
    property_area: str = Field(..., description="Property area (Urban/Semiurban/Rural)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LoanPrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    application: LoanApplication
    predicted_default_probability: float
    predicted_class: str
    risk_level: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DatasetStats(BaseModel):
    total_records: int
    default_rate: float
    avg_income: float
    avg_loan_amount: float
    missing_values: Dict[str, int]
    feature_distributions: Dict[str, Any]

# Sample dataset creation
def create_sample_dataset():
    """Create a realistic sample loan dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic loan data
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.exponential(5000, n_samples).astype(int),
        'CoapplicantIncome': np.random.exponential(2000, n_samples).astype(int),
        'LoanAmount': np.random.normal(150, 50, n_samples).astype(int),
        'Loan_Amount_Term': np.random.choice([360, 240, 180, 120], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'Credit_History': np.random.choice([1, 0], n_samples, p=[0.85, 0.15]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.35, 0.25])
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic default patterns
    default_prob = (
        0.1 +  # base probability
        (df['Credit_History'] == 0) * 0.4 +  # poor credit history increases risk
        (df['ApplicantIncome'] < 3000) * 0.2 +  # low income increases risk  
        (df['LoanAmount'] > 200) * 0.15 +  # high loan amount increases risk
        (df['Self_Employed'] == 'Yes') * 0.1 +  # self-employed increases risk
        np.random.normal(0, 0.1, n_samples)  # random noise
    )
    
    default_prob = np.clip(default_prob, 0, 0.8)  # cap at 80%
    df['Loan_Status'] = (np.random.random(n_samples) > default_prob).astype(int)
    df['Loan_Status'] = df['Loan_Status'].map({1: 'Y', 0: 'N'})  # Y = approved, N = default risk
    
    return df

# Load and train model
def train_model():
    """Train the logistic regression model on sample data"""
    global model, scaler, feature_columns
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Preprocessing
    df = df.copy()
    
    # Handle categorical variables
    le_dict = {}
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Prepare features and target
    feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                      'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    
    X = df[feature_columns]
    y = df['Loan_Status'].map({'Y': 0, 'N': 1})  # 0 = approved, 1 = default
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Model trained with accuracy: {accuracy:.3f}")
    
    # Save model, scaler and label encoders
    joblib.dump(model, ROOT_DIR / 'loan_model.pkl')
    joblib.dump(scaler, ROOT_DIR / 'scaler.pkl')
    joblib.dump(le_dict, ROOT_DIR / 'label_encoders.pkl')
    
    return df

# Initialize model on startup
sample_df = train_model()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Loan Default Prediction API", "status": "active"}

@api_router.get("/dataset/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """Get comprehensive dataset statistics"""
    try:
        df = sample_df.copy()
        
        # Basic stats
        total_records = len(df)
        default_rate = (df['Loan_Status'] == 'N').mean()
        avg_income = df['ApplicantIncome'].mean()
        avg_loan_amount = df['LoanAmount'].mean()
        
        # Missing values (simulated)
        missing_values = {col: 0 for col in df.columns}  # Sample data has no missing values
        
        # Feature distributions (using original string labels, convert numpy types to Python types)
        feature_distributions = {
            'income_distribution': {
                'low': int((df['ApplicantIncome'] < 3000).sum()),
                'medium': int(((df['ApplicantIncome'] >= 3000) & (df['ApplicantIncome'] < 7000)).sum()),
                'high': int((df['ApplicantIncome'] >= 7000).sum())
            },
            'loan_amount_distribution': {
                'small': int((df['LoanAmount'] < 100).sum()),
                'medium': int(((df['LoanAmount'] >= 100) & (df['LoanAmount'] < 200)).sum()),
                'large': int((df['LoanAmount'] >= 200).sum())
            },
            'credit_history_distribution': {
                '1': int((df['Credit_History'] == 1).sum()),
                '0': int((df['Credit_History'] == 0).sum())
            },
            'education_distribution': {
                'Graduate': int((df['Education'] == 'Graduate').sum()),
                'Not Graduate': int((df['Education'] == 'Not Graduate').sum())
            },
            'property_area_distribution': {
                'Urban': int((df['Property_Area'] == 'Urban').sum()),
                'Semiurban': int((df['Property_Area'] == 'Semiurban').sum()),
                'Rural': int((df['Property_Area'] == 'Rural').sum())
            },
            'default_by_credit_history': {
                '1': float(df[df['Credit_History'] == 1]['Loan_Status'].apply(lambda x: x == 'N').mean()),
                '0': float(df[df['Credit_History'] == 0]['Loan_Status'].apply(lambda x: x == 'N').mean())
            }
        }
        
        return DatasetStats(
            total_records=total_records,
            default_rate=default_rate,
            avg_income=avg_income,
            avg_loan_amount=avg_loan_amount,
            missing_values=missing_values,
            feature_distributions=feature_distributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating stats: {str(e)}")

@api_router.post("/predict", response_model=LoanPrediction)
async def predict_loan_default(application: LoanApplication):
    """Predict loan default probability for a new application"""
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Load label encoders
        le_dict = joblib.load(ROOT_DIR / 'label_encoders.pkl')
        
        # Prepare input data
        input_data = {
            'Gender': application.gender,
            'Married': application.married,
            'Dependents': application.dependents,
            'Education': application.education,
            'Self_Employed': application.self_employed,
            'ApplicantIncome': application.applicant_income,
            'CoapplicantIncome': application.coapplicant_income,
            'LoanAmount': application.loan_amount,
            'Loan_Amount_Term': application.loan_amount_term,
            'Credit_History': application.credit_history,
            'Property_Area': application.property_area
        }
        
        # Encode categorical variables
        for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            try:
                input_data[col] = le_dict[col].transform([input_data[col]])[0]
            except ValueError:
                # Handle unknown categories by using the most frequent class
                input_data[col] = 0
        
        # Create feature vector
        X_new = np.array([[input_data[col] for col in feature_columns]])
        
        # Scale features
        X_new_scaled = scaler.transform(X_new)
        
        # Make prediction
        default_prob = model.predict_proba(X_new_scaled)[0][1]  # Probability of default
        predicted_class = "Default Risk" if default_prob > 0.5 else "Approved"
        
        # Determine risk level  
        if default_prob < 0.3:
            risk_level = "Low"
        elif default_prob < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Create prediction object
        prediction = LoanPrediction(
            application=application,
            predicted_default_probability=float(default_prob),
            predicted_class=predicted_class,
            risk_level=risk_level
        )
        
        # Store prediction in database
        await db.loan_predictions.insert_one(prediction.dict())
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@api_router.get("/predictions/history", response_model=List[LoanPrediction])
async def get_prediction_history(limit: int = 50):
    """Get recent prediction history"""
    try:
        predictions = await db.loan_predictions.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return [LoanPrediction(**pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@api_router.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        df = sample_df.copy()
        
        # Prepare data for model evaluation using the same preprocessing as training
        le_dict = joblib.load(ROOT_DIR / 'label_encoders.pkl')
        
        # Apply label encoding with error handling for unseen labels
        for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            try:
                df[col] = le_dict[col].transform(df[col].astype(str))
            except ValueError as e:
                # Handle unseen labels by using the most frequent class
                most_frequent = le_dict[col].classes_[0]
                df[col] = df[col].apply(lambda x: most_frequent if x not in le_dict[col].classes_ else x)
                df[col] = le_dict[col].transform(df[col].astype(str))
        
        X = df[feature_columns]
        y = df['Loan_Status'].map({'Y': 0, 'N': 1})
        
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        
        # Feature importance (coefficients for logistic regression) - convert to native Python types
        feature_importance = {col: float(coef) for col, coef in zip(feature_columns, model.coef_[0])}
        
        return {
            "accuracy": float(accuracy),
            "total_samples": int(len(y)),
            "default_rate": float(y.mean()),
            "feature_importance": feature_importance,
            "model_type": "Logistic Regression"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()