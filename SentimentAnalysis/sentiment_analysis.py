"""Call the course sentiment model and return its label and score."""
import requests

URL = (
    "https://sn-watson-sentiment-bert.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/SentimentPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"
}

def sentiment_analyzer(text_to_analyse):
    """Predict sentiment, returning None fields for rejected text."""
    response = requests.post(
        URL, headers=HEADERS,
        json={"raw_document": {"text": text_to_analyse}}, timeout=30
    )
    if response.status_code in (400, 500):
        return {"label": None, "score": None}
    response.raise_for_status()
    sentiment = response.json()["documentSentiment"]
    return {"label": sentiment["label"], "score": sentiment["score"]}
