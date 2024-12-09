import numpy as np
import nltk
from sklearn.cluster import KMeans
import time
import logging
import gensim.downloader as api
from driver import Driver
from apiDriver import APIDriver


class ContextoSolver:
    def __init__(self, webDriver="Firefox", gameNumber=None):
        self.word_model = api.load('word2vec-google-news-300')
        self.max_guesses = 100
        self.guesses_dict = {}
        self.tried_words = set()
        self.failed_refinements = set()  # Track word combinations that led nowhere
        self.stuck_counter = 0  # Track how many times we've been stuck in refinement

        # updated to be able to use the API driver as well
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
        word = word.lower()
        return (word in self.english_words
                and len(word) > 2
                and word.isalpha()
                and word not in self.tried_words)

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
        best_words = [word for word, _ in sorted_guesses[:3]]
        worst_words = [word for word, _ in sorted_guesses[-3:]]

        try:
            similar_words = self.word_model.most_similar(
                positive=best_words,
                negative=worst_words,
                topn=topn * 2
            )
            valid_words = [(word, score) for word, score in similar_words
                           if self.is_valid_word(word)]
            return valid_words[:topn]

        except KeyError as e:
            logging.error(f"Error in directional guessing: {e}")
            return []

    def make_next_guess_refined(self, topn=100, score_threshold=100):
        best_score = min(self.guesses_dict.values()) if self.guesses_dict else float('inf')
        best_word = min(self.guesses_dict.items(), key=lambda x: x[1])[0] if self.guesses_dict else None
        candidates = []  # Initialize candidates here

        # If we've been stuck too long, temporarily exit refinement
        if self.stuck_counter > 5:
            logging.info("Been stuck in refinement too long, trying directional approach")
            self.stuck_counter = 0  # Reset counter
            return self.make_next_guess_directional(topn)

        # Ultra-fine refinement for very good scores
        if best_score <= 20:
            logging.info(f"Very close with word '{best_word}' (score: {best_score}). Using ultra-fine refinement.")

            # Get most recent guess
            recent_guesses = sorted([(word, score) for word, score in self.guesses_dict.items()],
                                    key=lambda x: len(self.tried_words - {x[0]}))

            if len(recent_guesses) >= 2:
                last_word, last_score = recent_guesses[-1]

                # If last guess was significantly worse
                if last_score > best_score * 2:
                    logging.info(f"Last guess '{last_word}' (score: {last_score}) was much worse. Reversing direction.")

                    try:
                        best_vector = self.word_model[best_word]
                        last_vector = self.word_model[last_word]
                        direction = best_vector - last_vector
                        candidates = []  # Reset candidates

                        # Try multiple small steps in opposite direction
                        for step in [0.05, 0.1, 0.15, 0.2, 0.25]:  # Added more steps
                            new_vector = best_vector + (direction * step)
                            new_vector = new_vector / np.linalg.norm(new_vector)
                            similar = self.word_model.similar_by_vector(new_vector, topn=20)
                            candidates.extend([(word, score) for word, score in similar
                                               if self.is_valid_word(word) and word not in self.failed_refinements])

                        if candidates:
                            return candidates[:topn]
                    except Exception as e:
                        logging.error(f"Error in direction reversal: {e}")

            # Fallback approach when stuck
            if not candidates:
                try:
                    logging.info("Trying semantic neighbors of best word")
                    neighbors = self.word_model.most_similar(best_word, topn=50)
                    candidates = [(word, score) for word, score in neighbors
                                  if self.is_valid_word(word) and word not in self.failed_refinements]
                    if candidates:
                        return candidates[:topn]
                except Exception as e:
                    logging.error(f"Error in neighbor search: {e}")

        # Regular refinement for good but not excellent scores
        good_words = [(word, score) for word, score in self.guesses_dict.items()
                      if score < score_threshold]

        if not good_words:
            return self.make_next_guess_directional(topn)

        good_words = sorted(good_words, key=lambda x: x[1])[:3]
        total_inverse_score = sum(1 / score for _, score in good_words)
        weights = [(word, (1 / score) / total_inverse_score) for word, score in good_words]

        logging.info(f"Using weighted refinement with words: {weights}")

        try:
            candidates = []  # Reset candidates
            steps = [0.1, 0.15, 0.2, 0.25, 0.3]

            for word, weight in weights:
                base_vector = self.word_model[word]
                weighted_steps = [step * weight for step in steps]

                for step in weighted_steps:
                    new_vector = base_vector + (base_vector * step)
                    new_vector = new_vector / np.linalg.norm(new_vector)

                    similar = self.word_model.similar_by_vector(new_vector, topn=20)
                    candidates.extend([(word, score) for word, score in similar
                                       if self.is_valid_word(word)])

                    inward_vector = base_vector - (base_vector * step)
                    inward_vector = inward_vector / np.linalg.norm(inward_vector)

                    similar = self.word_model.similar_by_vector(inward_vector, topn=20)
                    candidates.extend([(word, score) for word, score in similar
                                       if self.is_valid_word(word)])

            seen = {}
            for word, score in candidates:
                if word not in seen or score > seen[word]:
                    seen[word] = score

            unique_candidates = [(word, score) for word, score in seen.items()]
            unique_candidates.sort(key=lambda x: x[1], reverse=True)

            if not candidates:
                self.stuck_counter += 1
                logging.info(f"No candidates found, stuck counter: {self.stuck_counter}")
            else:
                self.stuck_counter = 0  # Reset when we find candidates

            return unique_candidates[:topn]

        except Exception as e:
            logging.error(f"Error in refined guessing: {e}")
            return []

    def play_game(self):
        initial_probes = ['thing', 'place', 'time', 'person', 'animal',
                          'nature', 'food', 'action', 'object']
        logging.info("Starting game with probe words: " + str(initial_probes))

        for guess in initial_probes:
            if guess.lower() not in self.tried_words:
                self.tried_words.add(guess.lower())
                word, score = self.driver.guessWord(guess)
                if word is not None and score is not None:
                    self.guesses_dict[word.lower()] = score
                    logging.info(f"Probe guess: {word} - Score: {score}")
                time.sleep(1)

        consecutive_failures = 0
        guess_count = len(self.guesses_dict)
        in_refinement_phase = False
        refinement_attempts = 0
        no_candidates_streak = 0  # Track consecutive "no candidates" results
        phase_switches = 0  # Track number of phase switches
        last_phase_switch_time = time.time()  # Track timing of phase switches

        while not self.driver.checkIfGameOver() and guess_count < self.max_guesses:
            current_time = time.time()
            current_best_score = min(self.guesses_dict.values()) if self.guesses_dict else float('inf')

            # Check for being stuck in a cycle
            if no_candidates_streak > 50:  # If we've failed to find candidates 50 times in a row
                logging.info("Detected being stuck with no candidates. Ending game.")
                break

            # Check for rapid phase switching
            if current_time - last_phase_switch_time < 5:  # If less than 5 seconds between phase switches
                phase_switches += 1
                if phase_switches > 10:  # If we've switched phases rapidly more than 10 times
                    logging.info("Detected rapid phase switching cycle. Ending game.")
                    break
            else:
                phase_switches = 0  # Reset if switches aren't rapid

            # Phase switching logic
            if current_best_score < 100 and not in_refinement_phase:
                logging.info(f"Switching to refinement phase! Best score: {current_best_score}")
                in_refinement_phase = True
                refinement_attempts = 0
                last_phase_switch_time = current_time

            if in_refinement_phase:
                refinement_attempts += 1
                if refinement_attempts > 20:
                    logging.info("Been in refinement phase too long, switching to directional")
                    in_refinement_phase = False
                    last_phase_switch_time = current_time
                    self.failed_refinements.update(word for word, _ in sorted(self.guesses_dict.items(),
                                                                              key=lambda x: x[1])[:3])
                    continue

                candidates = self.make_next_guess_refined(topn=100)
            else:
                candidates = self.make_next_guess_directional(topn=100)

            if not candidates:
                no_candidates_streak += 1
                logging.warning(f"No candidates available (streak: {no_candidates_streak})")
                consecutive_failures += 1
                if consecutive_failures > 5:
                    if in_refinement_phase:
                        in_refinement_phase = False
                        last_phase_switch_time = current_time
                        logging.info("Switching out of refinement phase due to consecutive failures")

                    # Use different fallback words based on best scores
                    if current_best_score < 200:
                        fallback_words = ["look", "watch", "view", "notice", "observe",
                                          "sense", "detect", "spot", "find", "perceive"]
                    else:
                        fallback_words = ["world", "life", "work", "system", "group",
                                          "nature", "form", "type", "kind", "state"]

                    candidates = [(w, 0) for w in fallback_words if w not in self.tried_words]
                if not candidates:
                    continue
            else:
                no_candidates_streak = 0  # Reset the streak when we find candidates

            next_guesses = self.make_diverse_guesses(candidates, n_clusters=3)

            valid_guess_found = False
            for guess in next_guesses:
                if guess.lower() not in self.tried_words:
                    self.tried_words.add(guess.lower())
                    word, score = self.driver.guessWord(guess)
                    if word is not None and score is not None:
                        self.guesses_dict[word.lower()] = score
                        logging.info(f"Guess #{guess_count + 1}: {word} - Score: {score} " +
                                     f"(Phase: {'Refinement' if in_refinement_phase else 'Directional'})")
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

    def log(self, logTitle, guesses):
        self.driver.log(logTitle, guesses)


def main():
    solver = ContextoSolver(webDriver="API")  # Changed to use API driver by default
    try:
        guesses = solver.play_game()
        print(f"Game finished in {guesses} guesses")
    except Exception as e:
        logging.error(f"Error during game: {e}")
    finally:
        solver.cleanup()
        solver.log("Game Run", guesses)


if __name__ == "__main__":
    main()
