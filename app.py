from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import pandas as pd
import io
import os
import logging
from collections import Counter
from fastapi.responses import JSONResponse

# Initialize logging
# Initialize logging to both console and file
# Initialize logging to both console and file
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler("app.log")  # Log to a file
                    ])

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

model_path = '5CD-AI/Vietnamese-Sentiment-visobert'

logger.info("Setting up sentiment analysis pipeline")
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=0)
# Request body schema for single text
class SentimentRequest(BaseModel):
    text: str
@app.post("/predict")
def predict(request: SentimentRequest):
    logger.info("Received prediction request for text: %s", request.text)
    result = sentiment_task(request.text)
    logger.info("Prediction result: %s", result)
    return {"sentiment": result}



@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
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
    df["label"] = df["text"].apply(lambda x: sentiment_task(x)[0]["label"])

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
