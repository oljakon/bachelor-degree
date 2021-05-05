from pymorphy2 import MorphAnalyzer

from n_grams import generate_n_grams, generate_pos_n_grams
from parse_text import split_text, lemmatize


def main():
    morph = MorphAnalyzer()
    parsed_text = split_text('./text.txt', morph)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_grams_array = generate_n_grams(lemmatized_text, 5)
    pos_n_grams_array = generate_pos_n_grams(n_grams_array, morph)
    print(pos_n_grams_array)


if __name__ == '__main__':
    main()