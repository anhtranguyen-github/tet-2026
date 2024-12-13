import logging
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib  # Prefer joblib for model loading
import io
import os

# Custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
import emoji
import numpy as np
import scipy.sparse as sp
from collections import Counter
from underthesea import word_tokenize, text_normalize
import pandas as pd
import io
import csv
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Define your custom transformers
class RemoveConsecutiveSpaces(BaseEstimator, TransformerMixin):
    def remove_consecutive_spaces(self, s):
        return ' '.join(s.split())

    def transform(self, x):
        return [self.remove_consecutive_spaces(s) for s in x]

    def fit(self, x, y=None):
        return self


class VietnameseTextNormalize(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [text_normalize(s) for s in x]

    def fit(self, x, y=None):
        return self


class VietnameseWordTokenizer(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [' '.join(word_tokenize(s)) for s in x]

    def fit(self, x, y=None):
        return self


class RemoveEmoji(BaseEstimator, TransformerMixin):
    def remove_emoji(self, s):
        return ''.join(c for c in s if c not in emoji.EMOJI_DATA)

    def transform(self, x):
        return [self.remove_emoji(s) for s in x]

    def fit(self, x, y=None):
        return self


# Initialize FastAPI app and logging
app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Allow your frontend port
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Globals for custom transformers to avoid serialization issues
globals().update({
    "RemoveConsecutiveSpaces": RemoveConsecutiveSpaces,
    "VietnameseTextNormalize": VietnameseTextNormalize,
    "VietnameseWordTokenizer": VietnameseWordTokenizer,
    "RemoveEmoji": RemoveEmoji,
})

# Load the sentiment model
sentiment_model = None
try:
    model_path = "new_model.pkl"  # Update with your model's path
    sentiment_model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load the model: %s", e)
    logger.error(traceback.format_exc())


# Define schemas
class SentimentRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: SentimentRequest):
    if sentiment_model is None:
        logger.error("Prediction failed: Model is not loaded.")
        return {"error": "Model is not loaded. Please contact the administrator."}

    try:
        prediction = sentiment_model.predict([request.text])
        result = prediction[0]
        logger.info("Prediction result: %s", result)
        return {"sentiment": result}
    except Exception as e:
        logger.error("Prediction error: %s", e)
        logger.error(traceback.format_exc())
        return {"error": "Failed to make prediction"}



@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    if sentiment_model is None:
        logger.error("Batch prediction failed: Model is not loaded.")
        return {"error": "Model is not loaded. Please contact the administrator."}

    logger.info("Received batch prediction request for file: %s", file.filename)

    # Check file type
    if file.content_type not in ["text/plain", "text/csv"]:
        logger.error("Unsupported file type: %s", file.content_type)
        return {"error": "Unsupported file type. Please upload a .txt or .csv file."}

    # Read file content
    content = file.file.read()

    # Handle .txt files
    if file.filename.endswith(".txt"):
        texts = content.decode("utf-8").splitlines()
        df = pd.DataFrame(texts, columns=["text"])

    # Handle .csv files
    elif file.filename.endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
            if "text" not in df.columns:
                logger.error("CSV file missing 'text' column")
                return {"error": "CSV file must contain a 'text' column."}
        except Exception as e:
            logger.exception("Failed to process CSV file")
            return {"error": f"Failed to process CSV file: {str(e)}"}

    else:
        logger.error("Unsupported file extension: %s", file.filename)
        return {"error": "Unsupported file extension. Only .txt and .csv are allowed."}

    # Perform sentiment analysis
    logger.info("Performing sentiment analysis on file: %s", file.filename)

    try:
        df["label"] = df["text"].apply(lambda x: sentiment_model.predict([x])[0])
    except Exception as e:
        logger.error("Error in applying prediction: %s", str(e))
        return {"error": f"Failed to apply predictions: {str(e)}"}

    # Calculate label ratios
    label_counts = Counter(df["label"])
    total_labels = sum(label_counts.values())
    label_ratios = {label: count / total_labels for label, count in label_counts.items()}

    # Save the result to a new CSV file
    output_filename = f"output_{file.filename.rsplit('.', 1)[0]}.csv"  # Use the base name of the input file
    output_path = os.path.join("./", output_filename)
    df.to_csv(output_path, index=False)

    logger.info("File processed successfully. Output saved to: %s", output_path)
    return JSONResponse(
        {
            "message": "File processed successfully.",
            "output_file": output_filename,
            "label_ratios": label_ratios
        }
    )