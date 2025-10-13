corpus = [
    "there is a big house",
    "i buy a house",
    "they buy the new house"
]

from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

from utils import *

word_frequencies = get_word_frequencies(corpus, tokenizer)
print(word_frequencies)

alphabet = get_and_sort_alphabet(word_frequencies)
print(alphabet)

init_vocab, splits = build_initial_vocabulary_and_splits(alphabet, word_frequencies)

pair_frequencies = compute_pair_frequencies(splits, word_frequencies)

# for i, key in enumerate(pair_frequencies.keys()):
#     print(f"{key}: {pair_frequencies[key]}")

print_best_pair(pair_frequencies)

vocab, merges = build_vocabulary_and_merges(splits, word_frequencies, init_vocab)
print(f"Vocabulary: {vocab}")
print(f"Merges: {merges}")

print(tokenize("they buy a bigger house", tokenizer, merges))
