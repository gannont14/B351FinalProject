from sklearn.feature_extraction.text import CountVectorizer
import nltk

text = ["The quick brown fox jumped over the lazy dog"]

vectorizer = CountVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)
