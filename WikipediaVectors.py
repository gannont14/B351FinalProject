import multiprocessing
from datasets import load_dataset
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk
import re


class WikipediaVectors:
    DATASET_DIR = "WikipediaCorpus"
    DATASET_NAME = "wikipedia"
    DATASET_CFG = "20220301.en"
    MODEL_PATH = "sklearn-track/wikiVectorModel.model"

    def __init__(self, ds_name=DATASET_NAME, ds_config=DATASET_CFG):
        import os
        nltk.download("punkt", quiet=True)

        self.model = None
        if os.path.exists(self.MODEL_PATH):
            # file is there, don't need to train it
            print("Loading existing model...")
            self.model = self.load()
        else:
            # If the file isn't there then train the new one
            print("Training new model...")
            self.dataset = load_dataset(ds_name, ds_config)
            self.min_word_len = 2
            self.max_word_len = 20
            self.model = self.train()
            self.save(self.model)

    def preprocess(self, text):
        text = text.lower()
        # remove special characters then tokenize
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = nltk.word_tokenize(text)
        # filter words
        words = [
            word for word in words
            if (self.min_word_len <= len(word) <= self.max_word_len)
        ]
        return words

    def text_generator(self):
        for shard_index in range(0, 41):
            shard = self.dataset['train'][shard_index]

            # should be in dict format
            if isinstance(shard, dict) and 'text' in shard:
                text = shard['text']
                preprocessed = self.preprocess(text)
                yield preprocessed
            # fallback to shard str format, tested on another ds, probably goes unused
            elif isinstance(shard, str):
                preprocessed = self.preprocess(shard)
                yield preprocessed

    def train(self, vector_size=100, window=5, min_count=5, workers=None):
        if workers is None:
            workers = max(1, multiprocessing.cpu_count() - 1)  # hehehehe

            phrases = Phrases(self.text_generator(), min_count=5)
            bigram = Phraser(phrases)

            sentences = [
                bigram[words] for words in self.text_generator()
            ]

            print("training vectors (Word2Vec)...")

            return Word2Vec(
                sentences=sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=10
            )

    def save(self, model, path=MODEL_PATH):
        model.save(path)
        print(f"Model saved to {path}")

    def load(self, path=MODEL_PATH):
        return Word2Vec.load(path)

    def explore(self, model, word, topn=10):
        try:
            return model.wv.most_similar(word, topn=topn)
        except KeyError:
            print(f"Word '{word}' not found in model's vocabulary")
            return []


# if __name__ == "__main__":
#     vectors = WikipediaVectors()
#
#     model = vectors.train()
#     vectors.save(model)
#
#     word = "Language"
#     word = word.lower()
#
#     similar_words = vectors.explore(
#         model,
#         word,
#         topn=20
#     )
#
#     print(f"\nWords most similar to '{word}':")
#     for word, similarity in similar_words:
#         print(f"{word}: {similarity}")
