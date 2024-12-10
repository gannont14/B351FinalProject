import os
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import logging
import gensim.downloader as api
from apiDriver import APIDriver
from driver import Driver
from WikipediaVectors import WikipediaVectors


class ContextoSolver:
    def __init__(self, webDriver="Firefox", gameNumber=None):
        vectors = WikipediaVectors()
        if vectors.model is None:
            raise RuntimeError("Failed to load or train Word2Vec model")
        self.word_model = vectors.model.wv
        print(f"Total Keys: {len(self.word_model.key_to_index)}")
        self.max_guesses = 150
        self.guesses_dict = {}
        self.tried_words = set()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.contexto_log_file = os.path.join(self.script_dir, "contexto_solver.log")
        self.api_log_file = os.path.join(self.script_dir, "API_results.log")
        self.api_driver_log_file = os.path.join(self.script_dir, "api_driver.log")
        self.game_number = gameNumber

        if webDriver == "API":
            print("Creating api driver")
            self.driver = APIDriver(gameNumber)
        else:
            self.driver = Driver(webDriver)

        if gameNumber is None:
            self.driver.selectGameByGameNumber()
        else:
            self.driver.selectGameByGameNumber(gameNumber)

        try:
            nltk.download('words', quiet=True)
            from nltk.corpus import words
            self.english_words = set(w.lower() for w in words.words())
        except Exception as e:
            logging.error(f"Error loading English words: {e}")
            self.english_words = set(api.load('models/word2vec-google-news-300').index_to_key[:50000])

        self.setup_logging()
        self.initialize_probes()

    def initialize_probes(self):
        """Define and manage multiple sets of probe words."""
        self.probe_sets = {
            'general': [
                'way',      # methods, paths, processes
                'time',     # temporal concepts, periods
                'life',     # living things, existence
                'mind',     # mental concepts, thoughts
                'work',     # actions, tasks, jobs
                'part',     # components, pieces
                'world'     # global concepts, spaces
            ],
            'abstract': [
                'form',     # shapes, structures, types
                'state',    # conditions, situations
                'power',    # strength, ability, control
                'sense',    # perception, meaning
                'point',    # locations, ideas, purpose
                'course',   # paths, progression
                'matter'    # substance, importance
            ],
            'conceptual': [
                'fact',     # truth, information
                'change',   # transformation, difference
                'place',    # location, position
                'rule',     # guidelines, principles
                'case',     # instances, situations
                'system',   # organization, structure
                'level'     # degree, position
            ]
        }

    def try_probe_set(self, probe_set, max_probes=5):
        """Try a specific set of probes and return the best score achieved."""
        best_score = float('inf')
        probes_used = 0

        for guess in probe_set:
            if probes_used >= max_probes:
                break

            if guess.lower() not in self.tried_words:
                self.tried_words.add(guess.lower())
                word, score = self.driver.guessWord(guess)

                if word is not None and score is not None:
                    self.guesses_dict[word.lower()] = score
                    logging.info(f"Probe guess: {word} - Score: {score}")
                    best_score = min(best_score, score)
                    probes_used += 1

                time.sleep(1)

        return best_score, probes_used

    def adaptive_probing(self):
        """Adaptively try different probe sets based on results."""
        # Try initial set
        current_set = 'general'
        best_score, probes_used = self.try_probe_set(self.probe_sets[current_set], max_probes=3)
        total_probes = probes_used

        logging.info(f"Initial probe set '{current_set}' best score: {best_score}")

        # If score isn't promising, try another set
        if best_score > 1000:  # Threshold for "not promising"
            remaining_sets = [set_name for set_name in self.probe_sets.keys() if set_name != current_set]
            for set_name in remaining_sets:
                if total_probes >= 7:  # Maximum total probes limit
                    break

                logging.info(f"Trying probe set '{set_name}'")
                score, probes = self.try_probe_set(self.probe_sets[set_name], max_probes=2)
                total_probes += probes

                if score < best_score:
                    best_score = score
                    current_set = set_name

                if best_score < 500:  # Good enough to proceed
                    break

        logging.info(f"Completed probing with best score: {best_score}")
        return best_score

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.contexto_log_file),
                logging.StreamHandler(),
            ],
        )
        print(f"Logging to: {self.contexto_log_file}")

        if not os.path.exists(self.api_driver_log_file):
            with open(self.api_driver_log_file, 'a') as f:
                f.write("API Driver Log Initialized\n")

        if not os.path.exists(self.api_log_file):
            with open(self.api_log_file, 'a') as f:
                f.write("Timestamp,GameNumber,GuessCount,WinningGuess\n")

    def calculate_word_weights(self, sorted_guesses, num_words, score_range, min_score, max_score, is_positive=True):
        weighted_words = []
        for word, score in sorted_guesses[:num_words] if is_positive else sorted_guesses[-num_words:]:
            if is_positive:
                # For positive words, higher weights for lower scores
                weight = 1 + (max_score - score) / score_range
                # Square the weight to emphasize better scores more
                weight = weight * weight
            else:
                # For negative words, lower weights and linear scaling
                weight = 1 + (score - min_score) / score_range

            # Add multiple copies based on weight
            weighted_words.extend([word] * int(weight * 3 if is_positive else weight))

        return weighted_words

    def make_next_guess_directional(self, topn=100):
        if len(self.guesses_dict) < 2:
            seeds = ['time', 'person', 'year', 'way', 'day']
            candidates = []
            for seed in seeds:
                try:
                    similar = self.word_model.most_similar(seed, topn=20)
                    candidates.extend([(word, score) for word, score in similar
                                       if self.is_valid_word(word)])
                except KeyError:
                    continue
            return candidates[:topn]

        sorted_guesses = sorted(self.guesses_dict.items(), key=lambda x: x[1])
        min_score = sorted_guesses[0][1]
        max_score = sorted_guesses[-1][1]
        score_range = max_score - min_score if max_score != min_score else 1

        # Calculate weighted words
        best_words_weighted = self.calculate_word_weights(
            sorted_guesses, 3, score_range, min_score, max_score, is_positive=True)
        worst_words_weighted = self.calculate_word_weights(
            sorted_guesses, 3, score_range, min_score, max_score, is_positive=False)

        try:
            similar_words = self.word_model.most_similar(
                positive=best_words_weighted,
                negative=worst_words_weighted,
                topn=topn * 2
            )
            return [(word, score) for word, score in similar_words
                    if self.is_valid_word(word)][:topn]

        except KeyError as e:
            logging.error(f"Error in directional guessing: {e}")
            return []

    def make_next_guess_refined(self, topn=100, score_threshold=100):
        """Refined guessing strategy for when we're getting closer to the target."""
        best_score = min(self.guesses_dict.values()) if self.guesses_dict else float('inf')
        best_word = min(self.guesses_dict.items(), key=lambda x: x[1])[0] if self.guesses_dict else None

        sorted_guesses = sorted(self.guesses_dict.items(), key=lambda x: x[1])
        min_score = sorted_guesses[0][1]
        max_score = sorted_guesses[-1][1]
        score_range = max_score - min_score if max_score != min_score else 1

        if best_score <= 50:
            logging.info(f"Very close with word '{best_word}' (score: {best_score}). Using fine refinement.")
            try:
                # Get weighted versions of nearby words
                best_words_weighted = self.calculate_word_weights(
                    sorted_guesses, 5, score_range, min_score, max_score, is_positive=True)

                candidates = []
                for word in best_words_weighted:
                    try:
                        similar = self.word_model.most_similar(word, topn=10)
                        candidates.extend(similar)
                    except KeyError:
                        continue

                # Deduplicate while keeping highest scores
                seen = {}
                for word, score in candidates:
                    if self.is_valid_word(word):
                        if word not in seen or score > seen[word]:
                            seen[word] = score

                unique_candidates = [(word, score) for word, score in seen.items()]
                unique_candidates.sort(key=lambda x: x[1], reverse=True)
                return unique_candidates[:topn]

            except Exception as e:
                logging.error(f"Error in fine refinement: {e}")
                return self.make_next_guess_directional(topn)

        # If score is good but not excellent, use weighted directional
        best_words_weighted = self.calculate_word_weights(
            sorted_guesses, 4, score_range, min_score, max_score, is_positive=True)
        worst_words_weighted = self.calculate_word_weights(
            sorted_guesses, 2, score_range, min_score, max_score, is_positive=False)

        try:
            similar_words = self.word_model.most_similar(
                positive=best_words_weighted,
                negative=worst_words_weighted,
                topn=topn * 2
            )
            return [(word, score) for word, score in similar_words
                    if self.is_valid_word(word)][:topn]
        except Exception as e:
            logging.error(f"Error in refined directional guessing: {e}")
            return self.make_next_guess_directional(topn)

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

    def is_valid_word(self, word):
        word = word.lower()
        bad_words = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                     "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
        return (word in self.english_words
                and word not in bad_words
                and len(word) > 2
                and word.isalpha()
                and word not in self.tried_words)

    def play_game(self):
        """Main game loop with adaptive probing."""
        logging.info("Starting game with adaptive probing")
        best_probe_score = self.adaptive_probing()

        consecutive_failures = 0
        guess_count = len(self.guesses_dict)
        in_refinement_phase = False

        try:
            while not self.driver.checkIfGameOver() and guess_count < self.max_guesses:
                current_best_score = min(self.guesses_dict.values()) if self.guesses_dict else float('inf')

                if current_best_score < 100 and not in_refinement_phase:
                    logging.info(f"Switching to refinement phase! Best score: {current_best_score}")
                    in_refinement_phase = True
                    candidates = self.make_next_guess_refined(topn=100)
                else:
                    logging.info("Using directional strategy")
                    candidates = self.make_next_guess_directional(topn=100)

                if not candidates:
                    logging.warning("No candidates available")
                    consecutive_failures += 1

                    if consecutive_failures >= 10:
                        logging.error("Too many consecutive failures. Exiting the game.")
                        raise RuntimeError("Too many consecutive failures")

                    if consecutive_failures > 5:
                        common_words = ["world", "life", "work", "system", "group"]
                        candidates = [(w, 0) for w in common_words if w not in self.tried_words]

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
                            logging.info(f"Guess #{guess_count + 1}: {word} - Score: {score} "
                                         + f"(Phase: {'Refinement' if in_refinement_phase else 'Directional'})")
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
                winning_guess = min(self.guesses_dict, key=self.guesses_dict.get)
                logging.info(f"Game completed in {guess_count} guesses! Winning guess: {winning_guess}")
                self.log_results(self.driver.gameNum, guess_count, winning_guess)
            else:
                logging.info(f"Game stopped after {guess_count} guesses without finding solution")
            return guess_count

        except RuntimeError as e:
            logging.error(f"Game terminated due to error: {e}")
            return -1

    def cleanup(self):
        self.driver.quitDriver()

    def log_results(self, game_number, guess_count, winning_guess):
        if not os.path.exists(self.api_log_file):
            with open(self.api_log_file, "w") as f:
                f.write("Timestamp,GameNumber,GuessCount,WinningGuess\n")

        with open(self.api_log_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{game_number},{guess_count},{winning_guess}\n")

    def log(self, logTitle, guesses):
        log_file = os.path.join(self.script_dir, "log.txt")
        with open(log_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Game #{self.game_number} - {guesses} guesses\n")
        logging.info(f"Game logged: {logTitle} - {guesses} guesses")


def main():
    solver = ContextoSolver(webDriver="API", gameNumber=805)
    try:
        guesses = solver.play_game()
        if guesses == -1:
            print("Game failed due to consecutive failures.")
        else:
            print(f"Game finished in {guesses} guesses")
    except Exception as e:
        logging.error(f"Error during game: {e}")
    finally:
        solver.cleanup()
        if guesses != -1:  # Log only if the game was not terminated due to error
            solver.log("Contexto solver", guesses)


if __name__ == "__main__":
    main()
