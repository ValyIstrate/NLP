from collections import defaultdict

def compute_pair_frequencies(splits, word_frequencies):
    pair_frequencies = defaultdict(int)
    for word, freq in word_frequencies.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_frequencies[pair] += freq
    return pair_frequencies

def merge_pair(a, b, splits, word_frequencies):
    for word in word_frequencies:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits

def get_word_frequencies(corpus, tokenizer):
    word_frequencies = defaultdict(int)

    for sentence in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_frequencies[word] += 1

    return word_frequencies

def get_and_sort_alphabet(word_frequencies):
    alphabet = []
    for word in word_frequencies.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    return alphabet

def print_best_pair(pair_frequencies):
    best_pair = ""
    max_freq = None

    for pair, freq in pair_frequencies.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    print(best_pair, max_freq)

def build_initial_vocabulary_and_splits(alphabet, word_frequencies):
    init_vocab = ["<|endoftext|>"] + alphabet.copy()
    splits = {word: [c for c in word] for word in word_frequencies.keys()}
    return init_vocab, splits

def build_vocabulary_and_merges(splits, word_frequencies, vocab):
    merges = {}

    while True:
        pair_frequencies = compute_pair_frequencies(splits, word_frequencies)

        if not pair_frequencies:
            print("No more pairs to merge — stopping.")
            break

        best_pair = max(pair_frequencies.items(), key=lambda x: x[1])[0]
        max_frequency = pair_frequencies[best_pair]

        if max_frequency == 0:
            print("All remaining pairs have frequency 0 — stopping.")
            break

        splits = merge_pair(best_pair[0], best_pair[1], splits, word_frequencies)

        merged_token = best_pair[0] + best_pair[1]
        merges[best_pair] = merged_token
        vocab.append(merged_token)

    return vocab, merges

def tokenize(txt, tokenizer, merges):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(txt)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
