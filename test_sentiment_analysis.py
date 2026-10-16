"""Check the three synthetic sentence-label pairs in the practice guide."""
import unittest
from SentimentAnalysis import sentiment_analyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Validate actual predictions from the course model."""

    def test_sentiment_analyzer(self):
        """Check positive, negative and neutral examples."""
        examples = {
            "I love working with Python": "SENT_POSITIVE",
            "I hate working with Python": "SENT_NEGATIVE",
            "I am neutral on Python": "SENT_NEUTRAL",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(sentiment_analyzer(text)["label"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
