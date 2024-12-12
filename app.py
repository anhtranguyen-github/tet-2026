from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import os
import logging
from collections import Counter
from fastapi.responses import JSONResponse
import joblib  # To load the .pkl model file

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler("app.log")  # Log to a file
                    ])

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load the model from the .pkl file
model_path = 'trained_model.pkl'  # Specify your .pkl file path here
logger.info("Loading machine learning model from %s", model_path)

# Initialize the sentiment_model as None
sentiment_model = None

try:
    sentiment_model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load the model: %s", str(e))

# Request body schema for single text
class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: SentimentRequest):
    if sentiment_model is None:
        logger.error("Prediction failed: Model is not loaded.")
        return {"error": "Model is not loaded. Please contact the administrator."}

    logger.info("Received prediction request for text: %s", request.text)
    try:
        # Use the loaded model to make a prediction
        prediction = sentiment_model.predict([request.text])
        result = prediction[0]  # Assuming the model returns an array with one element
        logger.info("Prediction result: %s", result)
        return {"sentiment": result}
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
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
