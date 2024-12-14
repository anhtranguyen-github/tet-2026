import pandas as pd
import re

# Read file into DataFrame
# Read file into DataFrame
df = pd.read_csv('comments.csv', names=['Comment'], encoding='utf-8', skip_blank_lines=True, on_bad_lines='skip')

# Drop rows with NaN or empty comments
df = df.dropna(subset=['Comment'])
df['Comment'] = df['Comment'].str.strip()
df = df[df['Comment'] != '']

# Define a function to clean the comments
def clean_comment(comment):
    # Remove rows containing https or http links
    if re.search(r'http[s]?://\S+', comment):
        return None
    # Remove rows containing phone numbers (sequences of 10+ digits or common phone patterns)
    if re.search(r'(\d{10,}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)', comment):
        return None
    # Remove rows with repetitive "sao kê"
    if re.search(r'(?:sao kê.*?){2,}', comment, flags=re.IGNORECASE):
        return None
    # Remove rows containing sequences of 4 or more digits
    if re.search(r'\d{4,}', comment):
        return None
    # Remove rows containing only double quotes
    if comment == '""':
        return None
    return comment.strip()

# Apply the cleaning function
df['Cleaned_Comment'] = df['Comment'].apply(clean_comment)

# Drop rows where the cleaned comment is None
df = df.dropna(subset=['Cleaned_Comment'])

# Drop comments with fewer than 4 words
df = df[df['Cleaned_Comment'].str.split().str.len() >= 4]

# Drop duplicate comments
df = df.drop_duplicates(subset=['Cleaned_Comment'])

# Save the cleaned comments to a new CSV file
df['Cleaned_Comment'].to_csv('cleaned_comments.csv', index=False, encoding='utf-8')

# Display the first few rows of the cleaned DataFrame
print(df.head())
