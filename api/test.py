# Data manipulation and utilities

from transformer_module import *
import joblib

# Load the trained model
clf = joblib.load('m3.pkl')
joblib.dump(clf, 'new_model.pkl') 
# Test loading
sample_text = ["This is a sample comment for testing."]
predictions = clf.predict(sample_text)

print(f"Predictions for sample text: {predictions}")


