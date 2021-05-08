from typing import List
from nltk import tokenize


def generate_n_grams(text: List[str], n: int) -> List[List[List]]:
    n_grams_array = []
    for sentence in text:
        tokens = tokenize.word_tokenize(sentence)
        n_grams = [tokens[i:i+n] for i in range(len(tokens) - n + 1)]
        if n_grams:
            n_grams_array.append(n_grams)

    return n_grams_array


def generate_pos_n_grams(n_grams_array: List[List[List]], morph) -> List[str]:
    pos_array = []
    for n_grams in n_grams_array:
        for n_gram in n_grams:
            n_pos_array = ''
            for token in n_gram:
                pos = morph.parse(token)[0].tag.POS
                n_pos_array += str(pos) + ' '
            n_pos_array = n_pos_array.rstrip()
            pos_array.append(n_pos_array)

    return pos_array
