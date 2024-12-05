import spacy
from transformers import BertTokenizer
import re
import json

# Load the BERT tokenizer and spaCy NLP model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_sm')

# Extract BERT vocabulary and filter subwords
bert_vocab = list(tokenizer.vocab.keys())
standalone_words = [word for word in bert_vocab if not word.startswith("##") and word.isalpha()]

# Function to filter English words
def is_english(word):
    return re.match(r'^[a-zA-Z]+$', word)

english_words = [word for word in standalone_words if is_english(word)]

# Function to filter nouns using spaCy
def filter_nouns(words):
    nouns = []
    for word in words:
        doc = nlp(word)
        if doc[0].pos_ == "NOUN":  # Check if the word is tagged as a noun
            nouns.append(word)
    return nouns

# Filter English words to include only nouns
noun_vocab = filter_nouns(english_words)

# Save noun vocabulary to a JSON file
output_file = "noun_vocab.json"
with open(output_file, "w") as f:
    json.dump(noun_vocab, f)

print(f"Filtered noun vocabulary saved to {output_file}")