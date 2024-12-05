import numpy as np
import nltk
from sklearn.cluster import KMeans
import time
import logging
from driver import Driver
import gensim.downloader as api


class ContextoSolver:
    def __init__(self, webDriver="Firefox", randomGame=False):
        # Use word2vec instead of GloVe
        self.word_model = api.load('word2vec-google-news-300')
        self.driver = Driver(webDriver)
        self.max_guesses = 300
        self.guesses_dict = {}
        self.tried_words = set()
        self.num_seeds = 3

        # Choose a random game
        if randomGame:
            self.driver.selectGameByGameNumber()

        # Download and load English word list
        try:
            nltk.download('words', quiet=True)
            from nltk.corpus import words
            self.english_words = set(w.lower() for w in words.words())
        except Exception as e:
            logging.error(f"Error loading English words: {e}")
            # Fallback to a basic common words list
            self.english_words = set(api.load('models/word2vec-google-news-300').index_to_key[:80000])

        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler('contexto_solver.log'),
                logging.StreamHandler()
            ]
        )

    def is_valid_word(self, word: str) -> bool:
        """Check if a word is a valid English word and meets our criteria."""
        word = word.lower()
        return (word in self.english_words and len(word) > 2 and word.isalpha() and word not in self.tried_words)

    def make_next_guess_with_negatives(self, topn=100):
        if len(self.guesses_dict) < 2:
            # Common English words as seeds
            seeds = ['time', 'person', 'year', 'way', 'day', 'thing', 'world', 'life', 'hand', 'part']
            candidates = []
            for seed in seeds:
                try:
                    similar = self.word_model.most_similar(seed, topn=20)
                    candidates.extend([(word, score) for word, score in similar
                                       if self.is_valid_word(word)])
                except KeyError:
                    continue
            return candidates[:topn]

        # Get highest and lowest scoring words
        sorted_guesses = sorted(self.guesses_dict.items(), key=lambda x: x[1], reverse=True)
        high_score_words = [word for word, score in sorted_guesses[:self.num_seeds]]
        low_score_words = [word for word, score in sorted_guesses[-self.num_seeds:]]

        logging.info(f"Using positive seeds: {high_score_words}")
        logging.info(f"Using negative seeds: {low_score_words}")

        try:
            similar_words = self.word_model.most_similar(
                positive=low_score_words,
                negative=high_score_words,
                topn=topn * 2  # Get more words since we'll filter some out
            )
            # Filter for valid English words
            valid_words = [(word, score) for word, score in similar_words
                           if self.is_valid_word(word)]
            return valid_words[:topn]
        except KeyError as e:
            logging.error(f"Word not in vocabulary: {e}")
            return []

    def make_diverse_guesses(self, candidates, n_clusters=5):
        if not candidates:
            return []

        if isinstance(candidates[0], tuple):
            words = [word for word, _ in candidates]
        else:
            words = candidates

        if len(words) < n_clusters:
            return words

        try:
            vectors = [self.word_model[word] for word in words if word in self.word_model]
            valid_words = [word for word in words if word in self.word_model]

            if not vectors:
                return words[:n_clusters]

            vectors = np.array(vectors)
            kmeans = KMeans(n_clusters=min(n_clusters, len(vectors)))
            clusters = kmeans.fit_predict(vectors)

            diverse_guesses = []
            for i in range(min(n_clusters, len(vectors))):
                cluster_words = [word for j, word in enumerate(valid_words)
                                 if clusters[j] == i]
                if cluster_words:
                    diverse_guesses.append(cluster_words[0])

            return diverse_guesses

        except Exception as e:
            logging.error(f"Error in clustering: {e}")
            return words[:n_clusters]

    def play_game(self):
        # Start with common English words
        initial_guesses = ["place", "thing", "idea"]
        logging.info("Starting game with initial guesses: " + str(initial_guesses))

        for guess in initial_guesses:
            if guess.lower() not in self.tried_words:
                self.tried_words.add(guess.lower())
                word, score = self.driver.guessWord(guess)
                if word is not None and score is not None:
                    self.guesses_dict[word.lower()] = score
                    logging.info(f"Initial guess: {word} - Score: {score}")
                time.sleep(1)

        consecutive_failures = 0
        guess_count = len(self.guesses_dict)

        while not self.driver.checkIfGameOver() and guess_count < self.max_guesses:
            topn = 100 + (consecutive_failures * 50)
            candidates = self.make_next_guess_with_negatives(topn=topn)

            if not candidates:
                logging.warning("No candidates available")
                consecutive_failures += 1
                if consecutive_failures > 5:
                    print("Failed too many times ----------------")
                    # Fallback to common English words
                    common_words = ["world", "life", "work", "place", "system",
                                    "group", "number", "point", "word", "state",
                                    "family", "fact", "head", "month", "book",
                                    "question", "story", "example", "house", "business"]
                    candidates = [(w, 0) for w in common_words if w not in self.tried_words]
                    # Change the number of seeds every time, can either double or just add, going to add first
                    self.num_seeds += 1
                    consecutive_failures = 0
                if not candidates:
                    continue

            next_guesses = self.make_diverse_guesses(candidates, n_clusters=3)

            valid_guess_found = False
            for guess in next_guesses:
                if guess.lower() not in self.tried_words:
                    self.tried_words.add(guess.lower())
                    word, score = self.driver.guessWord(guess)
                    if word is not None and score is not None:
                        self.guesses_dict[word.lower()] = score
                        logging.info(f"Guess #{guess_count + 1}: {word} - Score: {score}")
                        guess_count += 1
                        consecutive_failures = 0
                        valid_guess_found = True
                        break
                    else:
                        consecutive_failures += 1

            if not valid_guess_found:
                consecutive_failures += 1

            time.sleep(1)

        if self.driver.checkIfGameOver():
            logging.info(f"Game completed in {guess_count} guesses!")
        else:
            logging.info(f"Game stopped after {guess_count} guesses without finding solution")

        return guess_count

    def cleanup(self):
        self.driver.quitDriver()
