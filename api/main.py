import os
import sys
import pickle
import joblib
import numpy as np

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from api.schemas import PredictRequest, PredictResponse
from src.utils.text_processing import preprocess

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for Big Data Sentiment Analysis using XGBoost and BiLSTM.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 Templates
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
templates_dir = os.path.join(base_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Global variables for models
xgb_model = None
tfidf_vectorizer = None
bilstm_model = None
bilstm_tokenizer = None

# Constants
MAX_SEQUENCE_LENGTH = 100
LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.on_event("startup")
async def load_models():
    global xgb_model, tfidf_vectorizer, bilstm_model, bilstm_tokenizer
    
    models_dir = os.path.join(base_dir, "models")
    
    # Load XGBoost model
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    try:
        if os.path.exists(xgb_path) and os.path.exists(vec_path):
            xgb_model = joblib.load(xgb_path)
            tfidf_vectorizer = joblib.load(vec_path)
            print("XGBoost model loaded successfully.")
        else:
            print("XGBoost model files not found. Train the model first.")
    except Exception as e:
        print(f"Error loading XGBoost: {e}")

    # Load BiLSTM model
    bilstm_path = os.path.join(models_dir, "bilstm_model.h5")
    tok_path = os.path.join(models_dir, "bilstm_tokenizer.pkl")
    try:
        if os.path.exists(bilstm_path) and os.path.exists(tok_path):
            bilstm_model = load_model(bilstm_path)
            with open(tok_path, 'rb') as handle:
                bilstm_tokenizer = pickle.load(handle)
            print("BiLSTM model loaded successfully.")
        else:
            print("BiLSTM model files not found. Train the model first.")
    except Exception as e:
        print(f"Error loading BiLSTM: {e}")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/predict/xgboost", response_model=PredictResponse)
def predict_xgboost(request: PredictRequest):
    if xgb_model is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="XGBoost model is not loaded.")
    
    processed_text = preprocess(request.comment)
    if not processed_text:
        return PredictResponse(sentiment="neutral", confidence=0.0)
        
    vec_input = tfidf_vectorizer.transform([processed_text])
    
    # Predict probabilities
    probs = xgb_model.predict_proba(vec_input)[0]
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    
    sentiment = LABEL_MAP.get(pred_idx, "neutral")
    
    return PredictResponse(sentiment=sentiment, confidence=confidence)

@app.post("/api/predict/bilstm", response_model=PredictResponse)
def predict_bilstm(request: PredictRequest):
    if bilstm_model is None or bilstm_tokenizer is None:
        raise HTTPException(status_code=503, detail="BiLSTM model is not loaded.")
        
    processed_text = preprocess(request.comment)
    if not processed_text:
        return PredictResponse(sentiment="neutral", confidence=0.0)
        
    # Tokenize and pad
    seq = bilstm_tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Predict
    probs = bilstm_model.predict(padded_seq)[0]
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    
    sentiment = LABEL_MAP.get(pred_idx, "neutral")
    
    return PredictResponse(sentiment=sentiment, confidence=confidence)
