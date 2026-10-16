"""Serve the synthetic course sentiment-analysis demonstration."""
from flask import Flask, render_template, request
from SentimentAnalysis import sentiment_analyzer

app = Flask("Sentiment Analyzer")


@app.route("/")
def render_index_page():
    """Render the unmodified course interface."""
    return render_template("index.html")


@app.route("/sentimentAnalyzer")
def sent_analyzer():
    """Analyze the supplied text and format the service response."""
    result = sentiment_analyzer(request.args.get("textToAnalyze", ""))
    if result["label"] is None:
        return "Invalid input ! Try again."
    label = result["label"].removeprefix("SENT_")
    return f"The given text has been identified as {label} with a score of {result['score']}."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
