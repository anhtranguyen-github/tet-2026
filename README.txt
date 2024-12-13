
# FastAPI Sentiment Analysis Service Setup

Follow these steps to set up the FastAPI sentiment analysis service in a virtual environment.

## Step 1: Create a Python 3.11 Virtual Environment

To create a virtual environment using Python 3.11:

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to set up your project.
3. Run the following command to create a virtual environment:

   On Windows:
   ```
   python3.11 -m venv venv
   ```

   On macOS/Linux:
   ```
   python3.11 -m venv venv
   ```

4. Activate the virtual environment.

   On Windows:
   ```
   .\venv\Scripts\activate
   ```

   On macOS/Linux:
   ```
   source venv/bin/activate
   ```

## Step 2: Install Dependencies

Once the virtual environment is activated, you need to install the project dependencies.

1. Ensure that you have a `requirements.txt` file containing the necessary packages:
   ```
   fastapi
   uvicorn
   pandas
   joblib
   scikit-learn
   underthesea
   emoji
   ```

2. Install the dependencies by running:
   ```
   pip install -r requirements.txt
   ```

## Step 3: Run the FastAPI Application

After installing the dependencies, you can run the FastAPI application using `uvicorn`.

1. In the terminal, run the following command to start the server:
   ```
   uvicorn app:app --reload
   ```

2. The application will be running at `http://127.0.0.1:8000`. You can open this URL in your browser to interact with the API.

## Step 4: Interacting with the API

### `/predict` (POST)

This endpoint accepts a single text input and returns a sentiment prediction.

**Request Body:**
```json
{
  "text": "Your input text here"
}
```

**Response Example:**
```json
{
  "sentiment": 1
}
```
Where the sentiment can be:
- `0` = Negative
- `1` = Neutral
- `2` = Positive

### `/predict-batch` (POST)

This endpoint accepts a `.txt` or `.csv` file containing multiple text entries and returns sentiment predictions for each entry.

**Request Body:**
- `file`: A `.txt` or `.csv` file. The CSV file must contain a `text` column.

**Response Example:**
```json
{
  "message": "File processed successfully.",
  "output_file": "output_filename.csv",
  "label_ratios": {
    "0": 0.7,
    "1": 0.2,
    "2": 0.1
  }
}
```

This will process the file, return the output file name, and provide label ratios for the sentiment predictions.

