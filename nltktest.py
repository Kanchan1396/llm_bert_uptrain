from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import uptrain

from model_constants import *
from model_train import retrain_model
from helper_funcs import *

testing_texts = [
    "Nike shoes are very [MASK]."
]


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
original_model_outputs = [test_model(model, x) for x in testing_texts]


# Create Nike review training dataset
nike_attrs = {
    "version": "0.1.0",
    'source': "nike review dataset",
    'url': 'https://www.kaggle.com/datasets/tinkuzp23/nike-onlinestore-customer-reviews?resource=download',
}
# Download the dataset from the url, zip it and copy the csv file here
nike_reviews_dataset = create_dataset_from_csv("web_scraped.csv", "Content", "nike_reviews_data.json")


def nike_positive_sentiment_func(inputs, outputs, gts=None, extra_args={}):
    is_positives = []
    for input in inputs["text"]:
        txt = input.lower()
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(txt)

        is_negative = score['pos'] < 0.25
        for neg_adj in ['expensive', 'worn', 'cheap', 'inexpensive', 'dirty', 'bad']:
            if neg_adj in txt:
                is_negative = True

        is_positives.append(bool(1-is_negative))
    return is_positives

cfg = {
    'checks': [{
        'type': uptrain.Anomaly.EDGE_CASE,
        "signal_formulae": uptrain.Signal("Nike Positive Sentiment", nike_positive_sentiment_func)
    }],

    # Define where to save the retraining dataset
    'retraining_folder': "uptrain_smart_data",
    
    # Define when to retrain, define a large number because we are using UpTrain just to create retraining dataset
    'retrain_after': 10000000000
}

framework = uptrain.Framework(cfg)



with open(nike_reviews_dataset) as f:
    all_data = json.load(f)

for sample in all_data['data']:
    inputs = {'data': {'text': [sample['text']]}}
    #framework.log(inputs = inputs, outputs = None)
