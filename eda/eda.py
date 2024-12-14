import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the dataset
data = pd.read_csv('cleaned_labeled_data.csv')  # Updated to use labeled_high_score.csv

# Word Cloud for Visualizing Most Frequent Words
def generate_wordcloud(text_data, label_name):
    """Generate a word cloud for a specific label's text data."""
    text = ' '.join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Label: {label_name}', fontsize=16)
    filename = f'wordcloud_label_{label_name}.png'
    plt.savefig(filename)  # Save the figure as a file
    print(f"Saved word cloud for label '{label_name}' as '{filename}'")
    plt.close()  # Close the plot to avoid overlapping with next ones

# For Sentence Length Distribution
def plot_sentence_length_distribution(text_data):
    """Plot the distribution of sentence lengths in the dataset."""
    sentence_lengths = text_data.str.split().apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(sentence_lengths, bins=30, kde=False, color='blue')
    plt.title('Distribution of Sentence Lengths in Customer Reviews', fontsize=16)
    plt.xlabel('Sentence Length')
    plt.ylabel('Number of Sentences')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('sentence_length_distribution.png')  # Save the figure as a file
    print("Saved sentence length distribution as 'sentence_length_distribution.png'")
    plt.close()

# For Label Distribution
def plot_label_distribution(labels):
    """Plot the distribution of labels in the dataset."""
    label_counts = labels.value_counts()
    
    # Print the number of each label
    print("Label Counts:")
    print(label_counts)
    
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20c.colors)
    plt.title('Label Distribution in Dataset', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig('label_distribution.png')  # Save the figure as a file
    print("Saved label distribution as 'label_distribution.png'")
    plt.close()

# Main function to call all EDA components
def main():
    # Extract text and labels from the dataset
    X_data = data['Cleaned_Comment']  # Column name updated to 'Cleaned_Comment'
    y_data = data['Label']  # Column name updated to 'Label'

    # Generate word clouds for 3 distinct labels
    unique_labels = y_data.unique()
    print(f"Generating Word Clouds for {len(unique_labels)} labels: {unique_labels}")
    
    for label in unique_labels[:3]:  # Limit to first 3 labels
        label_text_data = X_data[y_data == label]  # Filter text data for the current label
        print(f"Generating Word Cloud for label '{label}'...")
        generate_wordcloud(label_text_data, label)

    # Plot sentence length distribution
    print("Plotting Sentence Length Distribution...")
    plot_sentence_length_distribution(X_data)

    # Plot label distribution
    print("Plotting Label Distribution...")
    plot_label_distribution(y_data)

if __name__ == '__main__':
    main()
