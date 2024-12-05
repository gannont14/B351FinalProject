import json
import time
from transformers import BertTokenizer, BertModel
from driver import Driver  # Import the Driver class
import torch
from scipy.spatial.distance import cosine

# Function to load the noun vocabulary
def load_noun_vocab(file_path):
    with open(file_path, "r") as f:
        noun_vocab = json.load(f)
    return noun_vocab

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings
def get_embedding(word):
    """
    Generate BERT embeddings for a given word with sentence context.
    """
    context_sentence = f"The word we are testing is {word}."  # Add context
    inputs = tokenizer(context_sentence, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embedding.detach().numpy()  # Detach from graph and convert to NumPY

# Function to get the next guess
def get_next_guess(recent_guess_embedding, candidate_words, guessed_words):
    """
    Find the most similar word to the recent guess using cosine similarity.
    """
    # Generate embeddings for all valid (not guessed) candidate words
    valid_candidates = {word: get_embedding(word) for word in candidate_words if word not in guessed_words}

    if not valid_candidates:
        print("No valid candidates remaining.")
        return None

    # Calculate cosine similarity for valid candidates
    similarities = {
        word: 1 - cosine(recent_guess_embedding.squeeze(), embedding.squeeze())
        for word, embedding in valid_candidates.items()
    }
    print(f"Similarity scores: {similarities}")

    # Select the word with the highest similarity
    next_guess = max(similarities, key=similarities.get)
    print(f"Selected next guess: {next_guess}")
    return next_guess

# Main function
if __name__ == "__main__":
    # Load the noun vocabulary from bert test.py
    noun_vocab_file = "noun_vocab.json"
    noun_vocab = load_noun_vocab(noun_vocab_file)
    print(f"Loaded noun vocabulary with {len(noun_vocab)} words.")

    # Initialize Selenium driver
    driver = Driver("Chrome")  # Change to your preferred browser
    driver.selectGameByGameNumber()  # Start a random game

    # Define initial candidate words and use noun_vocab
    candidate_words = noun_vocab[:50]  # Use the first 50 nouns as candidates (adjust as needed)

    guessed_words = set()  # Track guessed words

    try:
        # Start with the first guess
        current_guess = candidate_words[0]
        guessed_words.add(current_guess)
        feedback = driver.guessWord(current_guess)
        print(f"First guess: {current_guess}, Feedback: {feedback}")

        # Generate embeddings for the initial guess
        current_guess_embedding = get_embedding(current_guess)

        # Iterative guessing loop
        while not driver.checkIfGameOver():
            # Remove guessed words from candidates
            candidate_words = [word for word in candidate_words if word not in guessed_words]

            if not candidate_words:
                print("No candidate words remaining. Exiting.")
                break

            # Find the next best guess
            next_guess = get_next_guess(current_guess_embedding, candidate_words, guessed_words)

            if not next_guess:
                print("Failed to determine the next guess. Exiting.")
                break

            # Submit the guess to the Contexto game
            feedback = driver.guessWord(next_guess)

            # Handle invalid guesses
            if feedback[1] is None:
                print(f"Word '{next_guess}' is invalid. Removing from candidates.")
                guessed_words.add(next_guess)
                continue

            print(f"Guessed: {next_guess}, Feedback: {feedback}")

            # Wait to allow feedback to load
            time.sleep(2)  # Adjust the time (in seconds) if necessary

            # Update the guess embedding and guessed words
            current_guess_embedding = get_embedding(next_guess)
            guessed_words.add(next_guess)
            print(f"Guessed words: {guessed_words}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Quit the driver when finished
        driver.quitDriver()