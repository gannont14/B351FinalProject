import spacy
import json
import nltk
from nltk.corpus import words, brown, wordnet

# Download required datasets
nltk.download('words')
nltk.download('brown')
nltk.download('wordnet')

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Get English words and filter by common usage
raw_vocab = set(words.words())  # Raw list of all words
common_vocab = set(brown.words())  # More common words
english_words = list(raw_vocab.intersection(common_vocab))  # Only common words
total_words = len(english_words)


# Function to clean the vocabulary
def clean_vocab(vocab):
    """
    Clean the vocabulary by filtering out obscure, overly long, or non-alphabetic words.
    """
    return [
        word for word in vocab
        if len(word) >= 3 and len(word) <= 12 and word.islower() and word.isalpha()
    ]


# Function to validate nouns using WordNet
def is_valid_noun(word):
    """
    Validate that a word is a noun using WordNet.
    """
    synsets = wordnet.synsets(word)
    return any(synset.pos() == 'n' for synset in synsets)


# Function to filter nouns using spaCy
def filter_nouns(word_list):
    """
    Filters the given list of words to include only nouns using spaCy's POS tagging.
    """
    nouns = []
    for idx, word in enumerate(word_list):
        # Process each word with spaCy
        doc = nlp(word)
        # Check if the word is tagged as a noun
        if doc and doc[0].pos_ == "NOUN":
            nouns.append(word)

        # Print progress every 1000 words
        if idx % 1000 == 0 or idx == total_words - 1:
            print(f"Processed {idx + 1}/{total_words} words. Current noun count: {len(nouns)}")

    return nouns


# Step 1: Clean and filter the vocabulary
cleaned_vocab = clean_vocab(english_words)
print(f"Cleaned vocabulary size: {len(cleaned_vocab)}")

# Step 2: Validate nouns with WordNet
valid_vocab = [word for word in cleaned_vocab if is_valid_noun(word)]
print(f"Valid vocabulary size after WordNet filtering: {len(valid_vocab)}")

# Step 3: Filter the cleaned and valid vocabulary to include only nouns
print("Filtering nouns... This might take some time.")
noun_vocab = filter_nouns(valid_vocab)

# Save filtered nouns to a JSON file
output_file = "noun_vocab.json"
with open(output_file, "w") as f:
    json.dump(noun_vocab, f)

print(f"Filtered noun vocabulary saved to {output_file}")
print(f"Total nouns extracted: {len(noun_vocab)}")