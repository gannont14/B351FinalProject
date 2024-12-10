import nltk
from nltk.corpus import *

# List of datasets with "corpus" or "word" in their names
datasets = [
    "abc", "brown", "brown_tei", "cess_cat", "cess_esp",
    "comtrans", "conll2000", "conll2002", "crubadan", "europarl_raw",
    "genesis", "gutenberg", "inaugural", "jeita", "knbc", "mac_morpho",
    "machado", "masc_tagged", "ptb", "reuters", "semcor", "senseval",
    "shakespeare", "sinica_treebank", "state_union", "switchboard",
    "timit", "treebank", "webtext", "wordnet", "wordnet2021", "wordnet2022",
    "wordnet31", "words"
]


# Function to download and count words
def download_and_count_words(datasets):
    result = []
    for dataset in datasets:
        try:
            # Download the dataset
            print(f"Downloading dataset: {dataset}")
            nltk.download(dataset)

            # Access the dataset and check for .words() method
            corpus = getattr(nltk.corpus, dataset)
            if hasattr(corpus, 'words'):
                words = corpus.words()
                total_words = len(words)
                unique_words = len(set(words))
                result.append({
                    "Dataset": dataset,
                    "Total Words": total_words,
                    "Unique Words": unique_words
                })
            else:
                print(f"Dataset '{dataset}' does not support .words() method.")
        except Exception as e:
            print(f"Error accessing dataset '{dataset}': {e}")
    return result


# Get word counts for the datasets
word_counts = download_and_count_words(datasets)

# Print the results
print(f"{'Dataset':<20}{'Total Words':<15}{'Unique Words':<15}")
print("-" * 50)
for info in word_counts:
    print(f"{info['Dataset']:<20}{info['Total Words']:<15}{info['Unique Words']:<15}")