# Sentiment-Analysis-using-HuggingFace-BERT

This project performs sentiment analysis on tweets using a pre-trained Transformer model in PyTorch. It processes a dataset of tweets, predicts their sentiment labels (Positive, Negative, Neutral), and allows data shuffling for unbiased evaluation.

## Features
- Loads and cleans tweet dataset
- Applies a pre-trained Transformer model for sentiment classification
- Predicts sentiment labels as **Positive**, **Negative**, or **Neutral**
- Supports column display without truncation
- Shuffles the dataset for randomness
- Limits predictions to any number of rows (e.g., first 5 rows)

## Requirements
Before running the code, install the following dependencies:
```bash
pip install pandas torch transformers scikit-learn
How It Works
Load Dataset
The dataset is loaded into a pandas DataFrame and column names are standardized:

python
Copy
Edit
sentiment.columns = ['id', 'city', 'label', 'tweets']
Select Tweets Column
Extract only the tweets for sentiment prediction:

python
Copy
Edit
sentiments = sentiment[['tweets']]
Sentiment Scoring Function
Each tweet is tokenized and passed into the model to determine the sentiment:

python
Copy
Edit
def sentiment_score(tweet):
    tokens = tokenizer.encode(tweet, return_tensors='pt')
    results = model(tokens)
    sentiment = int(torch.argmax(results.logits)) + 1
    if sentiment in [1, 2]:
        return 'positive'
    elif sentiment in [4, 5]:
        return 'negative'
    else:
        return 'neutral'
Predict Labels for Selected Rows
For example, predicting only the first 5 rows:

python
Copy
Edit
sentiments = sentiments.head(5)
sentiments['label'] = sentiments['tweets'].apply(lambda x: sentiment_score(x[:512]))
Shuffle Data
Randomize the order of rows to avoid bias:

python
Copy
Edit
from sklearn.utils import shuffle
sentiments = shuffle(sentiments).reset_index(drop=True)
Display Full Tweets
Ensure tweets are printed fully without truncation:

python
Copy
Edit
pd.set_option('display.max_colwidth', None)
Example Output
tweets	label
I am coming to the borders and I will kill you...	negative
im getting on borderlands and i will kill you ...	negative
im coming on borderlands and i will murder you...	negative
im getting on borderlands 2 and i will murder ...	negative
im getting into borderlands and i can murder y...	negative

Notes
x[:512] is used to ensure the tweet text is within the model’s maximum token length limit.
