import multiprocessing
from datasets import load_dataset
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phrases, Phraser
import nltk
import re
from tqdm import tqdm
from typing_extensions import SupportsIndex


class WikipediaVectors:
    DATASET_DIR = "wikipedia"
    MODEL_PATH = "wikiVectorModel.model"

    def __init__(self, ds_name=DATASET_DIR):
        nltk.download("punkt", quiet=True)

        self.dataset = load_dataset(ds_name)

        self.min_word_len = 2
        self.max_word_len = 20
        self.model = self.convert_to_keyedvectors(self.load("wikiVectorModel.model"))

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
        # 12,9173,40
        num_shards = len(self.dataset['train'])
        subset = num_shards
        print(type(num_shards))
        for shard_index in tqdm(range(0, 64586), desc="Processing shards"):
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

    def train(self, vector_size=300, window=5, min_count=10, workers=None):
        if workers is None:
            workers = max(1, multiprocessing.cpu_count() - 1)  # hehehehe

            phrases = Phrases(self.text_generator(), min_count=5)
            bigram = Phraser(phrases)

            sentences = [
                bigram[words] for words in self.text_generator()
            ]

            print("training vectors (Word2Vec)...")

            with tqdm(total=10, desc="Training epochs", unit="epoch") as pbar:
                model1 =  Word2Vec(
                    sentences=sentences,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=workers,
                    epochs=10
                )
            return model1


    def save(self, model, path=MODEL_PATH):
        model.save(path)
        print(f"Model saved to {path}")

    def load(self, path=MODEL_PATH):
        return Word2Vec.load(path)

    def explore(self, model, positive=None, negative=None, topn=10):
        # Ensure positive and negative are lists, even if a single word is passed
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]

        # Default to empty lists if not provided
        positive = positive or []
        negative = negative or []

        if not positive and not negative:
            print("Error: At least one positive or negative word must be provided.")
            return []

        try:
            return model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        except KeyError as e:
            print(f"KeyError: {e}")
            return []

    def most_similar(self, positive, negative, topn):
        return self.explore(self.model, positive, negative, topn=topn)

    def convert_to_keyedvectors(self, model):
        keyed_vectors = KeyedVectors(vector_size=model.vector_size)
        # Get all words and their vectors from the model
        words = model.wv.index_to_key
        vectors = [model.wv[word] for word in words]

        # Add vectors to KeyedVectors object
        keyed_vectors.add_vectors(words, vectors)

        return keyed_vectors


if __name__ == "__main__":
    vectors = WikipediaVectors()

    model = vectors.train()
    vectors.save(model)

    positive = ["test"]
    negative = ["away"]
    positive = [word.lower() for word in positive]
    negative = [word.lower() for word in negative]

    similar_words = vectors.explore(
        model,
        positive,
        negative,
        topn=20
    )

    print(similar_words)
 #   for word, similarity in similar_words:
  #      print(f"{word}: {similarity}")
